import os
import argparse
import wandb
import torch
import cv2
import random
import pickle
from torchvision import utils
import numpy as np
import torch.optim as optim
from source import VanillaCVAE, ResNetCVAE
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from source.helpful_funs import simulate_dataset_with_outliers,  show_decoded_image, subset_29248images_from_30000images
from source.helpful_funs import generate_loss_latent_UMAP, IsolationForest_LocalOutlierFactor_OneClassSVM
from source.image_data_loader import ImageDataset, data_loader
from source.train import EarlyStopping, train
import itertools
from sklearn import preprocessing

def show_original_images(imageinfo_clinical, img_folderpath, batch_size, config, nrow, save_path):

    dataset = ImageDataset(csv_file=imageinfo_clinical, root_dir=img_folderpath, args=config)

    Dataset_DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    tk2 = tqdm(Dataset_DataLoader, total=int(len(Dataset_DataLoader)))
    for batch_idx, (data, target, index) in enumerate(tk2):
        out = utils.make_grid(data, nrow=nrow)

        if config.return_original_image_index:
            image_array = {'InputImage': out, 'original_index': index.tolist()}
        else:
            image_array = {'InputImage': out, 'data_index': index.tolist()}

        with open(save_path + '_batch{}.pkl'.format(batch_idx), 'wb') as f:
            pickle.dump(image_array, f)

def simulate_datasets_with_different_outlier_proportions(config, population_dataset=None, TrueOutliers_dataset=None):

    '''
    :param config: including outlier_props, outlier_types, outlier_numbers etc
    :param population_dataset: type: dataframe. for config.simulated_outlier_type in ['artificial'],
                               population_dataset is easy to obtain and is set to be None at first.
    :param TrueOutliers: type: dataframe.
    :return: None. Finish a simulation process
    '''

    if not os.path.exists('./data/simulation/{}/'.format(config.simulated_outlier_type)):
        os.makedirs('./data/simulation/{}/'.format(config.simulated_outlier_type))

    config.outlier_props = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    if config.simulated_outlier_type == 'artificial':
        config.outlier_list = ['black', 'white', 'contrast', 'gauss_noise', 'same_histogram']
    elif config.simulated_outlier_type == 'true':
        config.outlier_list = ['ellipse_medical_device', 'straight_medical_device', 'flip_half_breast', 'wrong_algorithm']
        population_dataset = population_dataset[~population_dataset.image_data_sha256.isin(list(TrueOutliers_dataset['image_data_sha256']))]
    elif config.simulated_outlier_type in ['data_augmentation', 'outlier_exposure']:
        population_dataset = population_dataset[~population_dataset.image_data_sha256.isin(list(TrueOutliers_dataset['image_data_sha256']))].copy()
        true_outliers_freq = TrueOutliers_dataset.groupby('inlier_outlier').agg(outlier_number=('inlier_outlier', 'count')).reset_index()
        config.outlier_list = list(true_outliers_freq['inlier_outlier'])

        if config.simulated_outlier_type in ['outlier_exposure']:
            config.outlier_props = [0.05, 0.1, 0.2]

    simulate_dataset_with_outliers(args=config, population_dataset=population_dataset, TrueOutliers_dataset=TrueOutliers_dataset)

def generate_dataset_for_outlier_exposure(imageinfo_clinical, true_outliers, outlier_score_df, train_valid_test_split, config):

    imageinfo_clinical['indices'] = list(range(imageinfo_clinical.shape[0]))
    imageinfo_clinical_datasplit = imageinfo_clinical.merge(train_valid_test_split, how='inner', on='indices')
    imageinfo_clinical_datasplit.drop(['indices'], axis=1, inplace=True)
    sha256_datasplit = imageinfo_clinical_datasplit.loc[:, ['image_data_sha256', 'train_valid_test']].copy()

    true_outliers_datasplit = true_outliers.merge(sha256_datasplit, how='inner', on='image_data_sha256')
    outlier_score_df_trainvalidtest = outlier_score_df.merge(sha256_datasplit, how='inner', on='image_data_sha256')

    outlier_score_df_train = outlier_score_df_trainvalidtest[outlier_score_df_trainvalidtest.train_valid_test.eq('train')]
    ensemble_outlier_scores_train = outlier_score_df_train.loc[:,['negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']]
    min_max_df = ensemble_outlier_scores_train.describe().loc[['min', 'max']]  # this is the statistics for the train dataset, and will be used for valid and test set

    outlier_score_df_trainvalid = outlier_score_df_trainvalidtest[outlier_score_df_trainvalidtest.train_valid_test.isin(['train', 'valid'])]
    if config.discovered_true_outlier_ratio == 'partial':
        ensemble_outlier_scores_trainvalid = outlier_score_df_trainvalid.loc[:,['image_data_sha256', 'negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']].copy()
        for column_name in ['negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']:
            min, max = list(min_max_df[column_name])[0], list(min_max_df[column_name])[1]
            ensemble_outlier_scores_trainvalid[column_name] = ensemble_outlier_scores_trainvalid[column_name].apply(lambda x: (x - min) / (max - min))
        ensemble_outlier_scores_trainvalid['ensemble1_average_scores'] = ensemble_outlier_scores_trainvalid[['negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']].mean(axis=1)
        ensemble_outlier_scores_trainvalid.sort_values(by=['ensemble1_average_scores'], ascending=True, inplace=True)
        ensemble_potential_outliers_1000 = ensemble_outlier_scores_trainvalid.iloc[0:1000, :]
        sha256_ensemble1000_list = list(ensemble_potential_outliers_1000['image_data_sha256'])
        true_outliers_datasplit['inlier_outlier_in_ensemble1000'] = true_outliers_datasplit['image_data_sha256'].apply(lambda x: -1 if x in sha256_ensemble1000_list else 1)
    elif config.discovered_true_outlier_ratio == 'all':
        #suppose all outliers in training and validation dataset have been discovered
        true_outliers_datasplit['inlier_outlier_in_ensemble1000'] = true_outliers_datasplit['image_data_sha256'].apply(lambda x: -1 if x in list(outlier_score_df_trainvalid['image_data_sha256']) else 1)

    true_outliers_in_ensemble1000 = true_outliers_datasplit[true_outliers_datasplit.inlier_outlier_in_ensemble1000.eq(-1)].copy()
    population_dataset = imageinfo_clinical_datasplit
    true_outliers_in_ensemble1000.drop(['pectoral_muscle', 'pectoral_muscle_labels', 'inlier_outlier_in_ensemble1000'], axis=1, inplace=True)

    simulate_datasets_with_different_outlier_proportions(config, population_dataset=population_dataset, TrueOutliers_dataset=true_outliers_in_ensemble1000)


def model_construct(config):
    if config.model == 'VanillaCVAE':
        # initialize the model, CMMD is grayscale, so image_channels is 1.
        model = VanillaCVAE(inputimage_width=config.resize_width, inputimage_height=config.resize_height, in_channels=config.image_channels,
                       first_out_channels=config.first_out_channels, latent_dim=config.latent_dim, hidden_layer_number=config.hidden_layer_number)
    elif config.model == 'ResNetCVAE':
        model = ResNetCVAE(num_Blocks=config.num_Blocks, z_dim=config.latent_dim, nc=config.image_channels, in_planes=config.first_out_channels,
                           image_height=config.resize_height, image_width=config.resize_width)

    return model

def model_train(config, model_path_dict, data_loader_path_dict):

    model = model_construct(config)
    model = model.to(config.device)

    if os.path.isfile(model_path_dict['load_model_path']):
        model.load_state_dict(torch.load(model_path_dict['load_model_path'], map_location=config.device))
    else:
        if config.tune_hyperparameter:
            wandb.watch(model, log="all")

        if config.train_method in ['VanillaCVAE_OE', 'VanillaCVAE_LOE']:
            img_folderpath = '/mnt/beegfs/mccarthy/scratch/projects/braix/'
            imageinfo_clinical_train = pd.read_excel('./data/simulation/outlier_exposure/{}/Hui_BRAIX_dataset_outlier_prop{}_train.xlsx'.format(config.discovered_true_outlier_ratio, config.outlier_prop))
            imageinfo_clinical_valid = pd.read_excel('./data/simulation/outlier_exposure/{}/Hui_BRAIX_dataset_outlier_prop{}_valid.xlsx'.format(config.discovered_true_outlier_ratio, config.outlier_prop))
            train_dataset = ImageDataset(csv_file=imageinfo_clinical_train, root_dir=img_folderpath, args=config)
            valid_dataset = ImageDataset(csv_file=imageinfo_clinical_valid, root_dir=img_folderpath, args=config)

            kwargs = config.kwargs
            trainloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **kwargs)
            validloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            trainloader, validloader, testloader = data_loader(config, data_loader_path_dict)

        if config.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, momentum=0.9)
        elif config.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, amsgrad=True)

        earlyStop = EarlyStopping(mode='min', patience=config.earlystop_patience, percentage=False)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=config.reduceLR_patience, threshold=1e-8, verbose=True)

        valid_loss_min = np.Inf # track change in validation loss
        pre_epochs = 100
        for epoch in range(1, config.epochs + 1):
            valid_loss_min = train(model, trainloader, validloader, optimizer, valid_loss_min, epoch, pre_epochs, config, model_path_dict['save_model_path'])

            scheduler.step(valid_loss_min)
            '''
            if earlyStop.step(valid_loss_min):
                print('\nEarly stop on epoch {}'.format(epoch))
                break
            '''
    # test_loss = test(model, config.device, testloader)

    return model

def generate_outlier_scores(model, config, outlier_scores_path):

    img_folderpath = '/mnt/beegfs/mccarthy/scratch/projects/braix/'
    #imageinfo_clinical = pd.read_excel(config.imageinfo_clinical_path)

    imageinfo_clinical = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')

    config.simulation = False
    dataset = ImageDataset(csv_file=imageinfo_clinical, root_dir=img_folderpath, args=config)
    Dataset_DataLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # get negative reconstruction loss, negative KL-Divergence loss, ELBO, latent z and UMAP for each image
    imageinfo_clinical_loss_latent_UMAP = generate_loss_latent_UMAP(Dataset_DataLoader, model, imageinfo_clinical, config.device)
    imageinfo_clinical_loss_latent_UMAP.to_excel(outlier_scores_path, header=True, index=False)

    '''    
    show_decoded_image(model=model, dataloader=Dataset_DataLoader, batches=1, mean=0, std=1, UMAP_df=None, 
    batch_size=16, config=config, save_dir = 'None', device=config.device)
    '''

    if config.find_potential_outliers_way in ['4_individual', 'ensemble1', 'ensemble2', 'ensemble3', 'ensemble4']:
        feature_type_list = ['latent']
    else:
        feature_type_list = ['latent', 'UMAP', 'negReconstLoss_latent', 'negReconstLoss_UMAP', 'negKLDLoss_latent', 'negKLDLoss_UMAP', 'ELBO_latent', 'ELBO_UMAP']

    if config.simulation and config.outlier_prop>0:
        outlier_prop = config.outlier_prop
    elif config.simulation and config.outlier_prop==0:
        outlier_prop = 0.005
    else:
        outlier_prop = 0.005

    for feature_type in feature_type_list:
        imageinfo_clinical_OutlierScores = IsolationForest_LocalOutlierFactor_OneClassSVM(imageinfo_clinical_loss_latent_UMAP, feature_type, outlier_prop)

    print('imageinfo_clinical_latent shape: {}'.format(imageinfo_clinical_OutlierScores.shape[1]))
    imageinfo_clinical_OutlierScores.to_excel(outlier_scores_path, header=True, index=False)

def map_SelectedPotentialOutliers_to_OriginalImageInfo(config):
    img_folderpath = '/mnt/beegfs/mccarthy/scratch/projects/braix/'

    config.imageinfo_clinical_path = './output/simulation/{}/imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_prop0.0.xlsx'.format(config.model)
    imageinfo_clinical = pd.read_excel(config.imageinfo_clinical_path)
    imageinfo_clinical['original_index'] = list(range(imageinfo_clinical.shape[0]))

    PotentialOutliers_info = pd.read_excel('./output/simulation/{}/PotentialOutliers/PotentialOutliers_info.xlsx'.format(config.model))
    outliers_OriginalIndex_list = list(set(list(PotentialOutliers_info['original_index'])))

    imageinfo_clinical_outliers = imageinfo_clinical[imageinfo_clinical.original_index.isin(outliers_OriginalIndex_list)]

    config.return_original_image_index = True

    imageinfo_clinical_outliers = pd.read_excel('./data/imageinfo_clinical_outliers.xlsx')
    imageinfo_clinical_outliers['inlier_outlier'] = 'inlier'
    outlier_dataset = ImageDataset(csv_file=imageinfo_clinical_outliers, root_dir=img_folderpath, args=config)

    Dataset_DataLoader = DataLoader(outlier_dataset, batch_size=100, shuffle=False, drop_last=False)
    tk2 = tqdm(Dataset_DataLoader, total=int(len(Dataset_DataLoader)))
    for batch_idx, (data, target, original_index) in enumerate(tk2):
        out = utils.make_grid(data, nrow=10)
        image_array = {'InputImage': out, 'original_index': original_index.tolist()}

        with open(config.save_dir + 'PotentialOutliers/TrueOutliers_batch{}.pkl'.format(batch_idx), 'wb') as f:
            pickle.dump(image_array, f)

        with open('./data/TrueOutliers_batch{}.pkl'.format(batch_idx), 'wb') as f:
            pickle.dump(image_array, f)

def select_PotentialOutliers(config, outlier_scores_path, selected_outliers_path):

    imageinfo_outlier_scores = pd.read_excel(outlier_scores_path)

    selected_number_list = [800, 400]
    selected_number_ensemble = selected_number_list[config.repetition_index]
    selected_number_individual = int(selected_number_ensemble/4)

    if config.find_potential_outliers_way in ['ensemble1_average', 'ensemble2_average', 'ensemble3_average', 'ensemble4_average']:
        ensemble_outlier_scores = imageinfo_outlier_scores.loc[:,['negReconstLoss', 'negKLDLoss', 'latent_LOF_scores', 'latent_OCSVM_scores']]
        min_max_scaler = preprocessing.MinMaxScaler()
        ensemble_outlier_scores_MinMax = min_max_scaler.fit_transform(ensemble_outlier_scores)
        imageinfo_outlier_scores['ensemble1_average_scores'] = list(np.mean(ensemble_outlier_scores_MinMax, axis=1))
        imageinfo_outlier_scores['ensemble2_average_scores'] = list(np.mean(ensemble_outlier_scores_MinMax[:, [0, 1]], axis=1))
        imageinfo_outlier_scores['ensemble3_average_scores'] = list(np.mean(ensemble_outlier_scores_MinMax[:, [0, 2]], axis=1))
        imageinfo_outlier_scores['ensemble4_average_scores'] = list(np.mean(ensemble_outlier_scores_MinMax[:, [0, 3]], axis=1))

    if config.find_potential_outliers_way == '4_individual':
        features_methods = list(itertools.product(['latent'], ['LOF', 'OCSVM']))
        features_methods += [('negReconstLoss', 'None'), ('negKLDLoss', 'None')]
    elif config.find_potential_outliers_way in ['ensemble1_average', 'ensemble2_average', 'ensemble3_average', 'ensemble4_average']:
        features_methods = [('{}'.format(config.find_potential_outliers_way), 'None')]
    else:
        features_methods = list(itertools.product(['latent', 'UMAP', 'negReconstLoss_latent', 'negReconstLoss_UMAP', 'negKLDLoss_latent', 'negKLDLoss_UMAP', 'ELBO_latent', 'ELBO_UMAP'], ['IF', 'LOF', 'OCSVM']))
        features_methods += [('negReconstLoss', 'None'), ('negKLDLoss', 'None'), ('ELBO', 'None')]

    if os.path.isfile(selected_outliers_path):
        imageinfo_clinical_PotentialOutliers = pd.read_excel(selected_outliers_path)
    else:
        imageinfo_clinical_PotentialOutliers = pd.DataFrame()

    new_discovered_outliers = pd.DataFrame()
    for feature, method in features_methods:
        if feature in ['negReconstLoss', 'negKLDLoss', 'ELBO']:
            predicted_scores_col = feature
        elif feature in ['ensemble1_average', 'ensemble2_average', 'ensemble3_average', 'ensemble4_average']:
            predicted_scores_col = '{}_scores'.format(feature)
        else:
            predicted_scores_col = '{}_{}_scores'.format(feature, method)

        imageinfo_outlier_scores_copy = imageinfo_outlier_scores.copy()

        imageinfo_outlier_scores_copy.sort_values(by=predicted_scores_col, ascending=True, inplace=True)

        if feature in ['ensemble1_average', 'ensemble2_average', 'ensemble3_average', 'ensemble4_average']:
            imageinfo_outlier_scores_copy_SmallestRows = imageinfo_outlier_scores_copy.iloc[0:selected_number_ensemble, :]
        else:
            imageinfo_outlier_scores_copy_SmallestRows = imageinfo_outlier_scores_copy.iloc[0:selected_number_individual, :]

        if new_discovered_outliers.shape[0] == 0:
            new_discovered_outliers = imageinfo_outlier_scores_copy_SmallestRows
        else:
            new_discovered_outliers = pd.concat([new_discovered_outliers, imageinfo_outlier_scores_copy_SmallestRows], axis=0)

    new_discovered_outliers.drop_duplicates(subset=['image_data_sha256'], inplace=True)
    new_discovered_outliers['repetition_index'] = config.repetition_index

    true_outliers_dataset_radiologist = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
    true_outliers_dataset = true_outliers_dataset_radiologist[true_outliers_dataset_radiologist.inlier_outlier_labels.eq(-1)]
    new_discovered_trueoutliers = (new_discovered_outliers[new_discovered_outliers.image_data_sha256.isin(list(true_outliers_dataset['image_data_sha256']))]).copy()

    if imageinfo_clinical_PotentialOutliers.shape[0]==0:
        imageinfo_clinical_PotentialOutliers = new_discovered_trueoutliers
    else:
        imageinfo_clinical_PotentialOutliers = pd.concat([imageinfo_clinical_PotentialOutliers, new_discovered_trueoutliers], axis=0)

    imageinfo_clinical_PotentialOutliers.to_excel(selected_outliers_path, header=True, index=False)

def model_DataLoader_path(config):

    if config.tune_hyperparameter:
        imageinfo_clinical_path = './data/Hui_BRAIX_dataset_info.xlsx'
        train_valid_test_split_index_path = 'None'
        selected_outliers_path = 'None'

        load_model_path = 'None'
        save_model_path = os.path.join(wandb.run.dir, 'model.pth')

    elif config.train_method in ['VanillaCVAE_ArtificialSimulation, VanillaCVAE_TrueSimulation, VanillaCVAE_AugmentationSimulation', 'VanillaCVAE_OE', 'VanillaCVAE_LOE']:
        if config.train_method in ['VanillaCVAE_OE', 'VanillaCVAE_LOE']:
            imageinfo_clinical_path = None #There are two paths, one for train dataset, one for valid dataset
        else:
            imageinfo_clinical_path = './data/simulation/{}/Hui_BRAIX_dataset_outlier_prop{}.xlsx'.format(config.simulated_outlier_type, config.outlier_prop)

        train_valid_test_split_index_path = os.path.join(config.save_dir, 'train_valid_test_split_prop{}.xlsx'.format(config.outlier_prop))
        selected_outliers_path = 'None'
        load_model_path = 'None'
        save_model_path = os.path.join(config.save_dir, 'model_prop{}.pth'.format(config.outlier_prop))
    elif config.recursive_outlier_detection:
        imageinfo_clinical_path = os.path.join(config.save_dir, 'imageinfo_clinical_Remove_PotentialOutiers_{}_{}.xlsx'.format(config.train_method, config.outlier_prop))
        selected_outliers_path = os.path.join(config.save_dir, 'imageinfo_clinical_PotentialOutliers_{}_{}.xlsx'.format(config.train_method, config.outlier_prop))
        train_valid_test_split_index_path = os.path.join(config.save_dir, 'train_valid_test_split_PotentialOutliers_{}_{}.xlsx'.format(config.train_method, config.outlier_prop))
        load_model_path = 'None'
        save_model_path = os.path.join(config.save_dir, 'model_PotentialOutliers_{}_{}.pth'.format(config.train_method, config.outlier_prop))

    data_loader_path_dict = {'imageinfo_clinical_path': imageinfo_clinical_path,
                             'selected_outliers_path': selected_outliers_path,
                             'train_valid_test_split_index_path': train_valid_test_split_index_path}
    model_path_dict = {'load_model_path': load_model_path, 'save_model_path': save_model_path}

    return model_path_dict, data_loader_path_dict

def optionFlags():
    parser = argparse.ArgumentParser(description='breast image')

    parser.add_argument('--taskid', type=int, default=0,
                        help='taskid from sbatch')

    parser.add_argument("--seed", type=int, default=42,
                      help="Defines the seed (default is 42)")

    parser.add_argument('--dataset_name', type=str, default='BRAIX',
                        help='the name of the dataset')

    parser.add_argument('--save_dir', type=str, default='./output/',
                        help='the directory to save results')

    parser.add_argument('--model', type=str, default='ResNetCVAE',
                        help='the model type')

    parser.add_argument('--tune_hyperparameter', action='store_true', default=False,
                        help='whether to tune hyperparameters')

    parser.add_argument('--simulation', action='store_true', default=False,
                        help='whether to run simulation study or not')

    parser.add_argument('--simulated_outlier_type', type=str, default='artificial',
                        help='the type of simulated outliers: artificial, true, data_augmentation, outlier_exposure')

    parser.add_argument('--discovered_true_outlier_ratio', type=str, default='partial',
                        help='whether use partial or all true outliers for the outlier exposure simulation')

    parser.add_argument('--stratify_split', action='store_true', default=False,
                        help='whether to split dataset with stratification')

    parser.add_argument('--use_crop', action='store_true', default=True,
                        help='whether to crop image')

    parser.add_argument('--return_original_image_index', action='store_true', default=False,
                        help='whether to return original image index')

    parser.add_argument('--outlier_prop', type=float, default=0,
                        help='the proportion of outliers in the simulation study')

    parser.add_argument('--imageinfo_clinical_path', type=str, default='./data/Hui_BRAIX_dataset_info.xlsx',
                        help='path for image metadata file')

    parser.add_argument('--recursive_outlier_detection', action='store_true', default=False,
                        help='whether to find potential outliers recursively in the BRAIx dataset')

    parser.add_argument('--train_method', type=str, default='VanillaCVAE_ArtificialSimulation',
                        help='VanillaCVAE_ArtificialSimulation, VanillaCVAE_TrueSimulation, VanillaCVAE_AugmentationSimulation, VanillaCVAE_OE, VanillaCVAE_LOE') #OE means outlier exposure

    parser.add_argument('--find_potential_outliers_way', type=str, default='27_individual',
                        help='which outlier scores to use to detect outliers')

    parser.add_argument('--repetition_times', type=int, default=3,
                        help='repeatition times for finding potential outliers')

    # for preprocessing
    parser.add_argument('--resize_height', type=int, default=256,
                        help='the size of resizing image')

    parser.add_argument('--resize_width', type=int, default=256,
                        help='the size of resizing image')

    # for ConvVAE
    parser.add_argument('--image_channels', type=int, default=1,
                        help='input image channels')

    parser.add_argument('--first_out_channels', type=int, default=8,
                        help='initial number of filters')

    parser.add_argument('--latent_dim', type=int, default=256,
                        help='latent dimension for sampling')

    parser.add_argument('--hidden_layer_number', type=int, default=5,
                        help='number of hidden layers in the encoder/decoder')

    parser.add_argument('--num_Blocks', type=list, default=[2,2,2,2],
                        help='number of blocks for each layer in encoder and decoder of ResNetCVAE')

    # for cross validation
    parser.add_argument('--cross_validation', action='store_true', default=False,
                        help='whether to do cross validation')

    parser.add_argument('--nfolds', type=int, default=3,
                        help='number of folds')

    # for training
    parser.add_argument('--train_set_ratio', type=float, default=0.6,
                        help='ratio of train set')

    parser.add_argument('--valid_set_ratio', type=float, default=0.1,
                        help='ratio of valid set')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')
    
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of data points per batch')

    parser.add_argument("--earlystop_patience", type=int, default=15,
                        help="Defines the early stop patience (default is 15)")

    parser.add_argument("--reduceLR_patience", type=int, default=5,
                        help="Defines the learning rate reduction patience (default is 5)")

    parser.add_argument("--scale", type=float, default=10,
                        help="scale for anomaly score")

    # general usage
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers to load data')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    config = parser.parse_args()

    return config


def main(config=None):

    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    config.device = torch.device("cuda" if use_cuda else "cpu")

    config.kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}

    cv2.setNumThreads(0)
    if config.device == torch.device('cuda'):
        print('use_cuda : {}\ndevice : {}\nGPU : {}'.format(use_cuda, config.device, torch.cuda.current_device()))

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.seed)  # python random seed
    np.random.seed(config.seed)  # numpy random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
          transforms.Resize((config.resize_height, config.resize_width)),
          transforms.ToTensor()
          #transforms.Normalize(mean=[0.1640], std=[0.1701])  # the mean and std are calculated for the BRAIx datasets
    ])

    config.transform = transform

    '''
    img_folderpath = '/mnt/beegfs/mccarthy/scratch/projects/braix/'
    imageinfo_clinical_potentialOutliers = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
    imageinfo_clinical = imageinfo_clinical_potentialOutliers[imageinfo_clinical_potentialOutliers.inlier_outlier_labels.eq(-1)]
    imageinfo_clinical = imageinfo_clinical.sort_values(by=['inlier_outlier'])
    dataset = ImageDataset(csv_file=imageinfo_clinical, root_dir=img_folderpath, args=config)
    save_path = './data/TrueOutliers'
    show_original_images(imageinfo_clinical, img_folderpath, 70, config, 10, save_path)
    '''

    if config.tune_hyperparameter:

        # os.environ['WANDB_MODE'] = 'dryrun'
        # os.environ['WANDB_START_METHOD'] = 'fork'

        # WandB – Initialize a new run
        '''
        wandb.init(project='cuan_2021_deeplearning-breastcancer', entity="huili", config=config)
        config = wandb.config
        wandb.run.name = wandb.run.id
        '''

        model_path_dict, data_loader_path_dict = model_DataLoader_path(config)
        model = model_train(config, model_path_dict, data_loader_path_dict)

        #show reconstructed output from trained model for hyperparameter tuning using wandb
        img_folderpath = '/mnt/beegfs/mccarthy/scratch/projects/braix/'
        imageinfo_clinical = pd.read_excel(config.imageinfo_clinical_path)
        if config.imageinfo_clinical_path == './data/Hui_BRAIX_dataset_info.xlsx':
            imageinfo_clinical = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')

        dataset = ImageDataset(csv_file=imageinfo_clinical, root_dir=img_folderpath, args=config)
        Dataset_DataLoader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
        show_decoded_image(model=model, dataloader=Dataset_DataLoader, batches=2, mean=0, std=1,
                           UMAP_df=None, batch_size=16, config=config, save_dir = wandb.run.dir, device=config.device)

    if config.simulation:

        if config.simulated_outlier_type == 'artificial':

            population_dataset = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')
            simulate_datasets_with_different_outlier_proportions(config, population_dataset=population_dataset)

        elif config.simulated_outlier_type in ['true', 'data_augmentation']:

            Potential_TrueOutlier = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
            TrueOutliers = Potential_TrueOutlier[Potential_TrueOutlier.inlier_outlier_labels.eq(-1)]
            population_dataset = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')

            simulate_datasets_with_different_outlier_proportions(config, population_dataset=population_dataset, TrueOutliers_dataset=TrueOutliers)

            '''
            # check the simulated outliers for the dataset with the largest outlier proportion
            img_folderpath = './data/simulation/outlier/'
            imageinfo_clinical = pd.read_excel('./data/simulation/Hui_BRAIX_dataset_outlier_prop0.1.xlsx')
            imageinfo_clinical_Outliers = imageinfo_clinical[~imageinfo_clinical.inlier_outlier.eq('inlier')]
            save_path = './data/simulation/outlier/SimulatedOutliers'
            show_original_images(imageinfo_clinical_Outliers, img_folderpath, 100, config, 10, save_path)
            '''

        elif config.simulated_outlier_type in ['outlier_exposure']:
            imageinfo_clinical = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')
            true_outliers_all = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
            true_outliers = true_outliers_all[true_outliers_all.inlier_outlier_labels.eq(-1)]
            train_valid_test_split = pd.read_excel('./output/simulation/VanillaCVAE/train_valid_test_split_prop0.0.xlsx')
            outlier_score_df = pd.read_excel('./output/simulation/VanillaCVAE/imageinfo_outlier_scores_prop0.0.xlsx')

            generate_dataset_for_outlier_exposure(imageinfo_clinical, true_outliers, outlier_score_df, train_valid_test_split, config)

    if config.train_method in ['VanillaCVAE, VanillaCVAE_ArtificialSimulation, VanillaCVAE_TrueSimulation, VanillaCVAE_AugmentationSimulation','VanillaCVAE_OE','VanillaCVAE_LOE']:

        if not os.path.exists('./output/{}/{}/'.format(config.train_method, config.discovered_true_outlier_ratio)):
            os.makedirs('./output/{}/{}/'.format(config.train_method, config.discovered_true_outlier_ratio))
        config.save_dir = './output/{}/{}/'.format(config.train_method, config.discovered_true_outlier_ratio)

        model_path_dict, data_loader_path_dict = model_DataLoader_path(config)

        model = model_train(config, model_path_dict, data_loader_path_dict)
        outlier_scores_path = os.path.join(config.save_dir, 'imageinfo_outlier_scores_prop{}.xlsx'.format(config.outlier_prop))
        generate_outlier_scores(model, config, outlier_scores_path)

    if config.recursive_outlier_detection:
        if not os.path.exists('./output/PotentialOutliers/'):
            os.makedirs('./output/PotentialOutliers/{}/')
        config.save_dir = './output/PotentialOutliers/{}/'

        config.repetition_index = 0
        model_path_dict, data_loader_path_dict = model_DataLoader_path(config)
        outlier_scores_path = os.path.join(config.save_dir, 'imageinfo_outlier_scores_PotentialOutliers_{}_{}.xlsx'.format(config.train_method, config.outlier_prop))

        while config.repetition_index < config.repetition_times:

            model = model_train(config, model_path_dict, data_loader_path_dict)
            generate_outlier_scores(model, config, outlier_scores_path)
            select_PotentialOutliers(config, outlier_scores_path, data_loader_path_dict['selected_outliers_path'])

            config.repetition_index += 1
if __name__ == '__main__':
    config = optionFlags()
    main(config)
