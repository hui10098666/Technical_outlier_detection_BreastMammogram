import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import math
import itertools
from sklearn import preprocessing
from source.helpful_funs import subset_29248images_from_30000images

def generate_inlier_outlier_labels(row):
    if row['inlier_outlier'] == 'inlier':
        return 1
    else:
        return -1

def generate_true_inlier_outlier_labels(row, TrueOutliers_sha256_list):
    if row['image_data_sha256'] in (TrueOutliers_sha256_list):
        return -1
    else:
        return 1

def confidence_interval(variable, confidence=0.95):
    ci_bottom = (100. - (confidence * 100.)) / 2.0
    ci_top = 100. - ci_bottom
    ci_bot_value, ci_top_value = list(map(int, np.percentile(variable, [ci_bottom, ci_top])))
    return ci_bot_value, ci_top_value


def bootstrap_dataset(config, df, outlier_type):

    if config.simulation:
        if config.simulated_outlier_type == 'artificial':
            outlier_type_list = ['black', 'white', 'contrast', 'gauss_noise', 'same_histogram']
        elif config.simulated_outlier_type == 'true':
            outlier_type_list = ['ellipse_medical_device', 'straight_medical_device', 'flip_half_breast', 'wrong_algorithm']
    else:
        df_outlier = df[df.inlier_outlier_labels.eq(-1)]
        outlier_type_list = list(set(list(df_outlier['inlier_outlier'])))

    df_inlier = df[df.inlier_outlier.eq('inlier')]

    if outlier_type == 'all':
        chosen_idx_inlier = np.random.choice(len(df_inlier), replace=True, size=len(df_inlier))
    else:
        if config.simulation:
            chosen_idx_inlier = np.random.choice(len(df_inlier), replace=True, size=int(len(df_inlier)/len(outlier_type_list)))
        else:
            df_OneOutlier = df[df.inlier_outlier.eq(outlier_type)]
            df_outlier = df[~df.inlier_outlier.eq('inlier')]
            OneOutlier_ratio = len(df_OneOutlier)/len(df_outlier)
            chosen_idx_inlier = np.random.choice(len(df_inlier), replace=True, size=int(len(df_inlier)*OneOutlier_ratio))
    df_trimmed = df_inlier.iloc[chosen_idx_inlier]

    if outlier_type == 'all':
        for (i, one_outlier_type) in enumerate(outlier_type_list):
            df_OneOutlier = df[df.inlier_outlier.eq(one_outlier_type)]
            if df_OneOutlier.shape[0] > 0:
                chosen_idx_OneOutlier = np.random.choice(len(df_OneOutlier), replace=True, size=len(df_OneOutlier))
                df_OneOutlier_trimmed = df_OneOutlier.iloc[chosen_idx_OneOutlier]
                df_trimmed = pd.concat([df_trimmed, df_OneOutlier_trimmed], axis=0)
    else:
        df_OneOutlier = df[df.inlier_outlier.eq(outlier_type)]
        chosen_idx_OneOutlier = np.random.choice(len(df_OneOutlier), replace=True, size=len(df_OneOutlier))
        df_OneOutlier_trimmed = df_OneOutlier.iloc[chosen_idx_OneOutlier]
        df_trimmed = pd.concat([df_trimmed, df_OneOutlier_trimmed], axis=0)

    return df_trimmed

def aucs(df, true_labels_col, predicted_scores_col):

    auroc = roc_auc_score(df[true_labels_col], df[predicted_scores_col])
    precision, recall, _ = precision_recall_curve(df[true_labels_col]*(-1), df[predicted_scores_col]*(-1))
    avgPrecision = average_precision_score(df[true_labels_col] * (-1), df[predicted_scores_col] * (-1))

    auprc = auc(recall, precision)

    #auroc_CI_lower, auroc_CI_upper = confidence_interval(aurocs, confidence=0.95)
    #auprc_CI_lower, auprc_CI_upper = confidence_interval(auprcs, confidence=0.95)
    #average_precision_score_CI_lower, average_precision_score_CI_upper = confidence_interval(average_precision_scores, confidence=0.95)

    #auroc_summary_dict = {'mean': np.mean(aurocs).round(2), 'std': np.std(aurocs).round(2), 'CI_lower': auroc_CI_lower, 'CI_upper': auroc_CI_upper}
    #auprc_summary_dict = {'mean': np.mean(auprcs).round(2), 'std': np.std(auprcs).round(2), 'CI_lower': auprc_CI_lower, 'CI_upper': auprc_CI_upper}
    #average_precision_score_summary_dict = {'mean': np.mean(average_precision_scores).round(2), 'std': np.std(average_precision_scores).round(2), 'CI_lower': average_precision_score_CI_lower, 'CI_upper': average_precision_score_CI_upper}

    return  auroc, auprc, avgPrecision

def F1_precision_recall(df, threshold_ratio, true_labels_col, predicted_scores_col):
    targets = df[true_labels_col]

    cut_off_value = list(np.percentile(df[predicted_scores_col], [threshold_ratio*100]))[0]
    preds = np.where(df[predicted_scores_col] < cut_off_value, -1, 1)

    cm1 = confusion_matrix(targets, preds)
    total1=sum(sum(cm1))

    TP = cm1[0, 0]
    TN = cm1[1, 1]
    FN = cm1[0, 1]
    FP = cm1[1, 0]

    if TP == 0 and FP == 0:
        precision = np.nan
    else:
        precision = TP/(TP + FP)

    if TP == 0 and FN == 0:
        recall = np.nan
    else:
        recall = TP/(TP + FN)

    if (precision==0 and recall == 0) or np.isnan(precision) or np.isnan(recall):
        F1_score =np.nan
    else:
        F1_score = 2*precision*recall/(precision + recall)

    # accuracy = (TP + TN) / total1
    # specificity = TN / (FP + TN)

    return precision, recall, F1_score

def get_ensemble_outlier_score(dataframe, min_max_df, ensembled_outlier_score_list):

    ensembled_outlier_score_list_renewed = ['image_data_sha256'] + ensembled_outlier_score_list
    dataframe_subset = dataframe.loc[:, ensembled_outlier_score_list_renewed].copy()
    for column_name in ensembled_outlier_score_list:
        min, max = list(min_max_df[column_name])[0], list(min_max_df[column_name])[1]
        dataframe_subset[column_name] = dataframe_subset[column_name].apply(lambda x: (x - min) / (max - min))
    dataframe_subset['ensemble1_average_scores'] = dataframe_subset[ensembled_outlier_score_list].mean(axis=1)

    return dataframe_subset

def optionFlags():
    parser = argparse.ArgumentParser(description='breast image')

    parser.add_argument('--taskid', type=int, default=0,
                        help='taskid from sbatch')

    parser.add_argument('--task', type=str, default='auroc_precision',
                        help='whether the task is to calculate auroc_precision or calculate recall rate in 200/600/1000 selected potential outliers')

    parser.add_argument("--seed", type=int, default=42,
                        help="Defines the seed (default is 42)")

    parser.add_argument('--dataset_name', type=str, default='BRAIX',
                        help='the name of the dataset')

    parser.add_argument('--save_dir', type=str, default='./output/simulation/',
                        help='the directory to save results')

    parser.add_argument('--model', type=str, default='VanillaCVAE',
                        help='the model type')

    parser.add_argument('--simulation', action='store_true', default=False,
                        help='summarize result for outliers in simulation study or not')

    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='whether to use ensemble of several outlier scores')

    parser.add_argument('--simulated_outlier_type', type=str, default='artificial',
                        help='the type of simulated outliers')

    parser.add_argument('--train_test', type=str, default='train',
                        help='train or test dataset')

    parser.add_argument('--n_boots', type=int, default=20,
                       help='number of bootstraps')

    # general usage
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers to load data')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    config = parser.parse_args()

    return config


def main(config=None):

    if config.task == 'auroc_precision':
        if config.simulation:
            config.outlier_props = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        else:
            config.outlier_props = [0.0]

        auroc_auprc_avgPrecision_TotalResult = pd.DataFrame(columns=['outlier_prop', 'outlier_type', 'feature', 'method', 'metric', 'mean', 'std'])
        precision_recall_F1score_TotalResult = pd.DataFrame(columns=['outlier_prop', 'outlier_type', 'feature', 'method', 'threshold_ratio', 'metric', 'mean', 'std'])

        for outlier_prop in config.outlier_props:
            imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_path = config.save_dir + '{}/imageinfo_outlier_scores_prop{}.xlsx'.format(config.model, outlier_prop)
            imageinfo_clinical_latent_UMAP_loss_OutlierMetrics = pd.read_excel(imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_path)

            if config.ensemble:
                ensemble_outlier_scores = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics.loc[:, ['negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']]
                min_max_scaler = preprocessing.MinMaxScaler()
                ensemble_outlier_scores_MinMax = min_max_scaler.fit_transform(ensemble_outlier_scores)
                imageinfo_clinical_latent_UMAP_loss_OutlierMetrics['ensemble1_scores'] = list(np.mean(ensemble_outlier_scores_MinMax, axis=1))
                imageinfo_clinical_latent_UMAP_loss_OutlierMetrics['ensemble2_scores'] = list(ensemble_outlier_scores_MinMax.min(axis=1))

            train_valid_test_split_path = config.save_dir + '{}/train_valid_test_split_prop{}.xlsx'.format(config.model, outlier_prop)
            train_valid_test_split = pd.read_excel(train_valid_test_split_path)
            #the imageinfo_clinical_latent_UMAP_loss_OutlierMetrics has the same row orders as the imageinfo_clinical subset with 29248 images
            imageinfo_clinical_latent_UMAP_loss_OutlierMetrics['indices'] = list(range(imageinfo_clinical_latent_UMAP_loss_OutlierMetrics.shape[0]))
            imageinfo_clinical_latent_UMAP_loss_OutlierMetrics = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics.merge(train_valid_test_split, how='inner', on='indices')
            imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics[imageinfo_clinical_latent_UMAP_loss_OutlierMetrics.train_valid_test.eq(config.train_test)].copy()

            if config.simulation:
                col = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.apply(lambda row: generate_inlier_outlier_labels(row), axis=1)  # get column data with an index
                dataframe = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.assign(inlier_outlier_labels=col.values)
            else:
                potential_outliers_dataset = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
                true_outliers_dataset = potential_outliers_dataset[potential_outliers_dataset.inlier_outlier_labels.eq(-1)]
                true_outliers_dataset_subset = true_outliers_dataset.loc[:,['image_data_sha256', 'inlier_outlier', 'inlier_outlier_labels']]

                if 'inlier_outlier' in imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.columns:
                    imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.drop(['inlier_outlier'], axis=1, inplace=True)
                if 'inlier_outlier_labels' in imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.columns:
                    imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.drop(['inlier_outlier_labels'], axis=1, inplace=True)

                dataframe_subset_inlier_slice = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset[~imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.image_data_sha256.isin(list(true_outliers_dataset_subset['image_data_sha256']))]
                dataframe_subset_inlier = dataframe_subset_inlier_slice.copy()
                dataframe_subset_inlier['inlier_outlier'] = 'inlier'
                dataframe_subset_inlier['inlier_outlier_labels'] = 1

                dataframe_subset_outlier = imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset[imageinfo_clinical_latent_UMAP_loss_OutlierMetrics_subset.image_data_sha256.isin(list(true_outliers_dataset_subset['image_data_sha256']))].copy()
                dataframe_subset_outlier = dataframe_subset_outlier.merge(true_outliers_dataset_subset, how='inner', on='image_data_sha256')

                dataframe = pd.concat([dataframe_subset_inlier, dataframe_subset_outlier], axis=0)

            outlier_list = list(set(list(dataframe['inlier_outlier'])))
            outlier_list.remove('inlier')
            config.outlier_types = ['all'] + outlier_list
            for outlier_type in config.outlier_types:
                if config.simulation:
                    if outlier_type == 'all':
                        dataframe_subset = dataframe.copy()
                    else:
                        dataframe_subset = dataframe[dataframe.inlier_outlier.isin(['inlier', outlier_type])]
                else:
                    dataframe_subset = dataframe.copy()

                if len(set(list(dataframe_subset['inlier_outlier_labels']))) == 1:
                    print('there is only one class in the test set')
                else:
                    features_methods = list(itertools.product(['latent', 'negReconstLoss_latent', 'negKLDLoss_latent','ELBO_latent'], ['IF', 'LOF', 'OCSVM']))
                    features_methods += [('negReconstLoss', 'None'), ('negKLDLoss', 'None'), ('ELBO', 'None')]
                    if config.ensemble:
                        features_methods += [('ensemble1', 'None'), ('ensemble2', 'None')]

                    auroc_auprc_avgPrecision = pd.DataFrame(columns=['feature', 'method', 'metric','value'])
                    precision_recall_F1score = pd.DataFrame(columns=['feature', 'method', 'threhold_ratio', 'metric', 'value'])
                    for boot_idx in range(config.n_boots):
                        df_trimmed = bootstrap_dataset(config=config, df=dataframe_subset, outlier_type=outlier_type)

                        for feature, method in features_methods:
                            if feature in ['negReconstLoss', 'negKLDLoss', 'ELBO']:
                                predicted_scores_col = feature
                            elif feature in ['ensemble1', 'ensemble2']:
                                predicted_scores_col = '{}_scores'.format(feature)
                            else:
                                predicted_scores_col = '{}_{}_scores'.format(feature, method)

                            auroc, auprc, avgPrecision = aucs(df=df_trimmed, true_labels_col='inlier_outlier_labels', predicted_scores_col=predicted_scores_col)

                            print('boot_idx: {}, feature: {}, method: {}, auroc: {}, auprc: {}, avgPrecision: {}'.format(boot_idx, feature, method, auroc, auprc, avgPrecision))

                            for metric, value in [('auroc', auroc), ('auprc', auprc), ('avgPrecision', avgPrecision)]:
                                auroc_auprc_avgPrecision = auroc_auprc_avgPrecision.append({'feature': feature, 'method': method, 'metric': metric, 'value': value}, ignore_index=True)

                            for threshold_ratio in [k / 10000 for k in (list(range(6, 20, 2)) + list(range(20, 80, 10)) + list(range(80, 1200, 20)) + list(range(1200, 3000, 200)))]:
                                precision, recall, F1_score = F1_precision_recall(df=df_trimmed, threshold_ratio=threshold_ratio, true_labels_col='inlier_outlier_labels', predicted_scores_col=predicted_scores_col)

                                print('boot_idx: {}, threshold_ratio: {}, feature: {}, method: {}, precision: {}, recall: {}, F1_score: {}'.format(boot_idx, threshold_ratio, feature, method, precision, recall, F1_score))
                                for metric, value in [('precision', precision), ('recall', recall), ('F1_score', F1_score)]:
                                    precision_recall_F1score = precision_recall_F1score.append({'feature': feature, 'method': method, 'threshold_ratio': threshold_ratio,'metric': metric, 'value': value}, ignore_index=True)

                    auroc_auprc_avgPrecision = auroc_auprc_avgPrecision[auroc_auprc_avgPrecision['value'].notna()]

                    auroc_auprc_avgPrecision_summary = auroc_auprc_avgPrecision.groupby(['feature','method', 'metric']).agg({'value':['mean','std']}).reset_index()
                    auroc_auprc_avgPrecision_summary.columns = ['feature', 'method', 'metric', 'mean', 'std']
                    auroc_auprc_avgPrecision_summary['outlier_prop'] = outlier_prop
                    auroc_auprc_avgPrecision_summary['outlier_type'] = outlier_type
                    auroc_auprc_avgPrecision_summary_copy = auroc_auprc_avgPrecision_summary.loc[:, ['outlier_prop', 'outlier_type', 'feature', 'method', 'metric', 'mean', 'std']]
                    if auroc_auprc_avgPrecision_TotalResult.shape[0] == 0:
                        auroc_auprc_avgPrecision_TotalResult = auroc_auprc_avgPrecision_summary_copy
                    else:
                        auroc_auprc_avgPrecision_TotalResult = pd.concat([auroc_auprc_avgPrecision_TotalResult, auroc_auprc_avgPrecision_summary_copy], axis=0)

                    precision_recall_F1score = precision_recall_F1score[precision_recall_F1score['value'].notna()]

                    precision_recall_F1score_summary = precision_recall_F1score.groupby(['feature', 'method', 'threshold_ratio', 'metric']).agg({'value':['mean','std']}).reset_index()
                    print(precision_recall_F1score_summary.columns)
                    print(precision_recall_F1score_summary)
                    precision_recall_F1score_summary.columns = ['feature', 'method', 'threshold_ratio', 'metric', 'mean', 'std']
                    precision_recall_F1score_summary['outlier_prop'] = outlier_prop
                    precision_recall_F1score_summary['outlier_type'] = outlier_type

                    precision_recall_F1score_summary_copy = precision_recall_F1score_summary.loc[:, ['outlier_prop', 'outlier_type', 'feature', 'method', 'threshold_ratio', 'metric', 'mean', 'std']]

                    if precision_recall_F1score_TotalResult.shape[0] == 0:
                        precision_recall_F1score_TotalResult = precision_recall_F1score_summary_copy
                    else:
                        precision_recall_F1score_TotalResult = pd.concat([precision_recall_F1score_TotalResult, precision_recall_F1score_summary_copy], axis=0)
            if config.ensemble:
                auroc_auprc_avgPrecision_TotalResult.to_excel(config.save_dir + '{}/auroc_auprc_avgPrecision_TotalResult_ensemble_{}.xlsx'.format(config.model, config.train_test), header=True, index=False)
                precision_recall_F1score_TotalResult.to_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_ensemble_{}.xlsx'.format(config.model, config.train_test), header=True, index=False)
            else:
                auroc_auprc_avgPrecision_TotalResult.to_excel(config.save_dir + '{}/auroc_auprc_avgPrecision_TotalResult_{}.xlsx'.format(config.model, config.train_test),header=True, index=False)
                precision_recall_F1score_TotalResult.to_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_{}.xlsx'.format(config.model, config.train_test),header=True, index=False)

        idx = precision_recall_F1score_TotalResult.groupby(['outlier_prop', 'outlier_type', 'feature', 'method', 'metric'])['mean'].transform(max) == precision_recall_F1score_TotalResult['mean']
        precision_recall_F1score_TotalResult_max_mean_rows = precision_recall_F1score_TotalResult[idx]
        precision_recall_F1score_TotalResult_max_mean_rows_noDuplicate = precision_recall_F1score_TotalResult_max_mean_rows.drop_duplicates(subset=['outlier_prop', 'outlier_type', 'feature', 'method', 'metric', 'mean'])

        if config.ensemble:
            precision_recall_F1score_TotalResult_max_mean_rows_noDuplicate.to_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_max_mean_ensemble_{}.xlsx'.format(config.model, config.train_test), header=True, index=False)
        else:
            precision_recall_F1score_TotalResult_max_mean_rows_noDuplicate.to_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_max_mean_{}.xlsx'.format(config.model, config.train_test),header=True, index=False)

    elif config.task == 'recall':

        potential_outliers_dataset = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
        true_outliers_dataset = potential_outliers_dataset[potential_outliers_dataset.inlier_outlier_labels.eq(-1)]
        train_valid_test_split = pd.read_excel('./output/simulation/VanillaCVAE/train_valid_test_split_prop0.0.xlsx')
        imageinfo_clinical = subset_29248images_from_30000images('./data/Hui_BRAIX_dataset_info.xlsx')

        imageinfo_clinical['indices'] = list(range(imageinfo_clinical.shape[0]))
        imageinfo_clinical_datasplit = imageinfo_clinical.merge(train_valid_test_split, how='inner', on='indices')
        imageinfo_clinical_datasplit.drop(['indices'], axis=1, inplace=True)
        sha256_datasplit = imageinfo_clinical_datasplit.loc[:, ['image_data_sha256', 'train_valid_test']].copy()

        true_outliers_inlier_outlier_label = true_outliers_dataset.loc[:,['image_data_sha256', 'inlier_outlier', 'inlier_outlier_labels', 'pectoral_muscle', 'pectoral_muscle_labels']]
        sha256_datasplit_outlier_labels = sha256_datasplit.merge(true_outliers_inlier_outlier_label, on='image_data_sha256', how='inner')
        sha256_datasplit_inlier_labels = sha256_datasplit[~(sha256_datasplit.image_data_sha256.isin(list(sha256_datasplit_outlier_labels['image_data_sha256'])))].copy()
        sha256_datasplit_inlier_labels['inlier_outlier'] = 'inlier'
        sha256_datasplit_inlier_labels['inlier_outlier_labels'] = 1
        sha256_datasplit_inlier_labels['pectoral_muscle'] = 'pectoral_muscle - no outliers'
        sha256_datasplit_inlier_labels['pectoral_muscle_labels'] = 1
        sha256_datasplit = pd.concat([sha256_datasplit_inlier_labels, sha256_datasplit_outlier_labels], axis=0)

        #generate 20 MCs for 20 bootstrap to compare VanillaCVAE, VanillaCVAE_LOE, VanillaCVAE_OE
        if not os.path.exists('./data/VanillaCVAE_LOE_OE/'):
            os.makedirs('./data/VanillaCVAE_LOE_OE/')
        '''
        if len(os.listdir('./data/VanillaCVAE_LOE_OE/')) == 0:
            for n_boot in range(config.n_boots):
                for data_type in ['train','test']:
                    if n_boot == 0:
                        sha256_datasplit_nobootstrap = sha256_datasplit[sha256_datasplit.train_valid_test.eq(data_type)]
                        sha256_datasplit_nobootstrap.to_excel('./data/VanillaCVAE_LOE_OE/' + 'sha256_datasplit_inlieroutlierinfo_{}_MC{}.xlsx'.format(data_type, n_boot))
                    else:
                        dataset_for_bootstrap = sha256_datasplit[sha256_datasplit.train_valid_test.eq(data_type)]
                        #in sha256_datasplit_bootstrapped, there could be duplicated rows.
                        sha256_datasplit_bootstrapped = bootstrap_dataset(config=config, df=dataset_for_bootstrap,outlier_type='all')
                        sha256_datasplit_bootstrapped.to_excel('./data/VanillaCVAE_LOE_OE/' + 'sha256_datasplit_inlieroutlierinfo_{}_MC{}.xlsx'.format(data_type, n_boot))
        '''
        #if select 200/600 potential outliers
        erosion_result_df1_train = pd.read_excel('./output/threshold_erosion/find_potential_outliers_by_threshold_erosion_train_taskid4.xlsx')
        erosion_result_df1_test = pd.read_excel('./output/threshold_erosion/find_potential_outliers_by_threshold_erosion_test_taskid4.xlsx')
        erosion_result_df1 = pd.concat([erosion_result_df1_train, erosion_result_df1_test], axis=0)

        #if select 1000 potential outliers
        erosion_result_df2_train = pd.read_excel('./output/threshold_erosion/find_potential_outliers_by_threshold_erosion_train_taskid0.xlsx')
        erosion_result_df2_test = pd.read_excel('./output/threshold_erosion/find_potential_outliers_by_threshold_erosion_test_taskid0.xlsx')
        erosion_result_df2 = pd.concat([erosion_result_df2_train, erosion_result_df2_test], axis=0)

        #if select 200 potential outliers
        muscle_cut_result_df1_train = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_train_taskid10.xlsx')
        muscle_cut_result_df1_test = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_test_taskid10.xlsx')
        muscle_cut_result_df1 = pd.concat([muscle_cut_result_df1_train, muscle_cut_result_df1_test], axis=0)


        # if select 600 potential outliers
        muscle_cut_result_df2_train = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_train_taskid7.xlsx')
        muscle_cut_result_df2_test = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_test_taskid7.xlsx')
        muscle_cut_result_df2 = pd.concat([muscle_cut_result_df2_train, muscle_cut_result_df2_test], axis=0)

        #if select 1000 potential outliers
        muscle_cut_result_df3_train = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_train_taskid9.xlsx')
        muscle_cut_result_df3_test = pd.read_excel('./output/muscle_cut/find_potential_outliers_by_muscle_cut_test_taskid9.xlsx')
        muscle_cut_result_df3 = pd.concat([muscle_cut_result_df3_train, muscle_cut_result_df3_test], axis=0)

        recall_overlap_outlier_df = pd.DataFrame()
        latent_learnt_prop = [('VanillaCVAE', 0.0)]

        for latent_learnt, prop in latent_learnt_prop:
            if latent_learnt == 'VanillaCVAE':
                outlier_score_path = './output/simulation/VanillaCVAE/imageinfo_outlier_scores_prop0.0.xlsx'
            else:
                outlier_score_path = './output/{}/imageinfo_outlier_scores_prop{}.xlsx'.format(latent_learnt, prop)

            outlier_score_df = pd.read_excel(outlier_score_path)
            if 'inlier_outlier' in outlier_score_df.columns:
               outlier_score_df.drop(['inlier_outlier'], axis=1, inplace=True)
            if 'inlier_outlier_labels' in outlier_score_df.columns:
               outlier_score_df.drop(['inlier_outlier_labels'], axis=1, inplace=True)

            #get ensembled outlier score using min and max statistics from train set during training
            outlier_score_df_trainvalidtest = outlier_score_df.merge(sha256_datasplit, how='inner', on='image_data_sha256')
            outlier_score_df_train = outlier_score_df_trainvalidtest[outlier_score_df_trainvalidtest.train_valid_test.eq('train')]
            ensembled_outlier_score_list = ['negReconstLoss', 'negKLDLoss', 'latent_OCSVM_scores']
            min_max_df = outlier_score_df_train.loc[:,ensembled_outlier_score_list].describe().loc[['min', 'max']]
            ensemble_outlier_score_train = get_ensemble_outlier_score(dataframe=outlier_score_df_train, min_max_df=min_max_df, ensembled_outlier_score_list=ensembled_outlier_score_list)
            outlier_score_df_test = outlier_score_df_trainvalidtest[outlier_score_df_trainvalidtest.train_valid_test.eq('test')]
            ensemble_outlier_score_test = get_ensemble_outlier_score(dataframe=outlier_score_df_test, min_max_df=min_max_df, ensembled_outlier_score_list=ensembled_outlier_score_list)

            ensemble_outlier_score_total_df = pd.concat([ensemble_outlier_score_train, ensemble_outlier_score_test], axis=0)

            for n_boot in range(config.n_boots):
                print('latent_learnt_method: {}, prop: {}, boot_idx: {}'.format(latent_learnt, prop, n_boot))
                result_OneConfig_OneMC = {'MC': [n_boot], 'latent_learnt_method': [latent_learnt], 'outlier_prop': [prop]}

                for data_type in ['train','test']:
                    sha256_datasplit_inlieroutlierinfo_df = pd.read_excel('./data/VanillaCVAE_LOE_OE/' + 'sha256_datasplit_inlieroutlierinfo_{}_MC{}.xlsx'.format(data_type, n_boot))
                    true_outlier_df_deeplearning = sha256_datasplit_inlieroutlierinfo_df[sha256_datasplit_inlieroutlierinfo_df.image_data_sha256.isin(list(true_outliers_dataset['image_data_sha256']))]
                    true_outlier_df_erosion = true_outlier_df_deeplearning[(~true_outlier_df_deeplearning.inlier_outlier.isin(['bad_positioning - outliers'])) & true_outlier_df_deeplearning.pectoral_muscle_labels.eq(1)]
                    true_outlier_df_muscle = true_outlier_df_deeplearning[true_outlier_df_deeplearning.pectoral_muscle_labels.eq(-1)]

                    result_OneConfig_OneMC.update({'reference_deeplearning_{}'.format(data_type): [true_outlier_df_deeplearning.shape[0]]})

                    '''
                    result_OneConfig_OneMC.update({'reference_deeplearning_{}'.format(data_type): [true_outlier_df_deeplearning.shape[0]],
                                                   'reference_erosion_{}'.format(data_type): [true_outlier_df_erosion.shape[0]],
                                                   'reference_muscle_cut_{}'.format(data_type): [true_outlier_df_muscle.shape[0]]})
                    '''

                    sha256_datasplit_inlieroutlierinfo_ensemble_outlierscore = sha256_datasplit_inlieroutlierinfo_df.merge(ensemble_outlier_score_total_df, on='image_data_sha256', how='left')
                    sha256_datasplit_inlieroutlierinfo_ensemble_outlierscore.sort_values(by=['ensemble1_average_scores'], ascending=True,inplace=True)

                    for selection_rate in [1, 2, 5]:
                        number = math.ceil(sha256_datasplit_inlieroutlierinfo_df.shape[0]*selection_rate/100)
                        ensemble_potential_outliers_subset = sha256_datasplit_inlieroutlierinfo_ensemble_outlierscore.iloc[0:number, :]
                        deeplearning_true_outliers = ensemble_potential_outliers_subset[ensemble_potential_outliers_subset.image_data_sha256.isin(list(true_outliers_dataset['image_data_sha256']))]
                        print(set(list(deeplearning_true_outliers['inlier_outlier'])))
                        #result_OneConfig_OneMC.update({'deeplearning_{}_{}'.format(number, data_type): [deeplearning_true_outliers.shape[0]]})
                        result_OneConfig_OneMC.update({'deeplearning_{}percent_percent_{}'.format(selection_rate, data_type): [deeplearning_true_outliers.shape[0]/true_outlier_df_deeplearning.shape[0]]})

                        if selection_rate in [1, 2]:
                            erosion_df_subset = erosion_result_df1.loc[:, ['image_data_sha256','image_array_sum_after_processing']]
                        else:
                            erosion_df_subset = erosion_result_df2.loc[:, ['image_data_sha256', 'image_array_sum_after_processing']]

                        sha256_datasplit_inlieroutlierinfo_erosion = sha256_datasplit_inlieroutlierinfo_df.merge(erosion_df_subset, on='image_data_sha256', how='left')
                        sha256_datasplit_inlieroutlierinfo_erosion.sort_values(by=['image_array_sum_after_processing'], ascending=False, inplace=True)
                        erosion_potential_outliers = sha256_datasplit_inlieroutlierinfo_erosion.iloc[0:number, :]
                        erosion_true_outliers = erosion_potential_outliers[erosion_potential_outliers.image_data_sha256.isin(list(true_outliers_dataset['image_data_sha256']))]
                        #result_OneConfig_OneMC.update({'erosion_{}_{}'.format(number, data_type): [erosion_true_outliers.shape[0]]})
                        #result_OneConfig_OneMC.update({'erosion_{}_percent_{}'.format(number, data_type): [erosion_true_outliers.shape[0]/true_outlier_df_erosion.shape[0]]})

                        deeplearning_true_outliers_list = list(deeplearning_true_outliers['image_data_sha256'])
                        erosion_true_outliers_list = list(erosion_true_outliers['image_data_sha256'])
                        increase_number_by_erosion = sum([0 if k in deeplearning_true_outliers_list else 1 for k in erosion_true_outliers_list])
                        deeplearning_erosion = len(deeplearning_true_outliers_list) + increase_number_by_erosion
                        #result_OneConfig_OneMC.update({'increase_by_erosion_{}_{}'.format(number, data_type): [increase_number_by_erosion]})
                        #result_OneConfig_OneMC.update({'increase_by_erosion_{}percent_percent_{}'.format(selection_rate, data_type): [increase_number_by_erosion/true_outlier_df_deeplearning.shape[0]]})
                        result_OneConfig_OneMC.update({'union_erosion_{}percent_percent_{}'.format(selection_rate, data_type): [deeplearning_erosion / true_outlier_df_deeplearning.shape[0]]})

                        if selection_rate == 1:
                            muscle_cut_df_subset = muscle_cut_result_df1.loc[:, ['image_data_sha256','line_number_in_muscle']]
                        elif selection_rate == 2:
                            muscle_cut_df_subset = muscle_cut_result_df2.loc[:,['image_data_sha256', 'line_number_in_muscle']]
                        else:
                            muscle_cut_df_subset = muscle_cut_result_df3.loc[:,['image_data_sha256', 'line_number_in_muscle']]

                        sha256_datasplit_inlieroutlierinfo_muscle_cut = sha256_datasplit_inlieroutlierinfo_df.merge(muscle_cut_df_subset, on='image_data_sha256', how='left')
                        sha256_datasplit_inlieroutlierinfo_muscle_cut_subset = sha256_datasplit_inlieroutlierinfo_muscle_cut[(sha256_datasplit_inlieroutlierinfo_muscle_cut['line_number_in_muscle'] <= 8) & (sha256_datasplit_inlieroutlierinfo_muscle_cut['line_number_in_muscle'] > 0)]
                        sha256_datasplit_inlieroutlierinfo_muscle_cut = sha256_datasplit_inlieroutlierinfo_muscle_cut_subset.copy()
                        sha256_datasplit_inlieroutlierinfo_muscle_cut.sort_values(by=['line_number_in_muscle'], ascending=False, inplace=True)
                        muscle_cut_potential_outliers = sha256_datasplit_inlieroutlierinfo_muscle_cut.iloc[0:number, :]
                        muscle_cut_true_outliers = muscle_cut_potential_outliers[muscle_cut_potential_outliers.image_data_sha256.isin(list(true_outliers_dataset['image_data_sha256']))]
                        #result_OneConfig_OneMC.update({'muscle_cut_{}_{}'.format(number, data_type): [muscle_cut_true_outliers.shape[0]]})
                        #result_OneConfig_OneMC.update({'muscle_cut_{}_percent_{}'.format(number, data_type): [muscle_cut_true_outliers.shape[0]/true_outlier_df_muscle.shape[0]]})

                        muscle_cut_true_outliers_list = list(muscle_cut_true_outliers['image_data_sha256'])
                        increase_number_by_muscle_cut = sum([1 if (k not in deeplearning_true_outliers_list) and (k not in erosion_true_outliers_list) else 0 for k in muscle_cut_true_outliers_list])
                        deeplearning_erosion_muscle_cut =  deeplearning_erosion + increase_number_by_muscle_cut

                        #result_OneConfig_OneMC.update({'increase_by_muscle_cut_{}_{}'.format(number, data_type): [increase_number_by_muscle_cut]})
                        #result_OneConfig_OneMC.update({'increase_by_muscle_cut_{}percent_percent_{}'.format(selection_rate, data_type): [increase_number_by_muscle_cut/true_outlier_df_deeplearning.shape[0]]})
                        result_OneConfig_OneMC.update({'union_msucle_cut_{}percent_percent_{}'.format(selection_rate, data_type): [deeplearning_erosion_muscle_cut / true_outlier_df_deeplearning.shape[0]]})
                        '''
                        total_outlier_number = len(deeplearning_true_outliers_list + [k for k in erosion_true_outliers_list if k not in deeplearning_true_outliers_list] +
                                                   [k for k in muscle_cut_true_outliers_list if (k not in deeplearning_true_outliers_list) and (k not in erosion_true_outliers_list)])
                        #result_OneConfig_OneMC.update({'deeplearning_erosion_muscle_cut_{}_{}'.format(number, data_type): [total_outlier_number]})
                        result_OneConfig_OneMC.update({'deeplearning_erosion_muscle_cut_{}percent_percent_{}'.format(selection_rate, data_type): [total_outlier_number/true_outlier_df_deeplearning.shape[0]]})
                        '''
                if recall_overlap_outlier_df.shape[0]==0:
                    recall_overlap_outlier_df = pd.DataFrame.from_dict(result_OneConfig_OneMC)
                else:
                    recall_overlap_outlier_df = pd.concat([recall_overlap_outlier_df, pd.DataFrame.from_dict(result_OneConfig_OneMC)], axis=0)

                recall_overlap_outlier_df.to_excel('./output/VanillaCVAE_VanillaCVAELOE_VanillaCVAEOE_erosion_muscle_cut_total_result.xlsx', header=True, index=False)

        recall_results_mean_std = pd.DataFrame()
        latent_prop_list = ['latent_learnt_method', 'outlier_prop']
        for i in range(len(recall_overlap_outlier_df.columns)-3):
            i += 3
            column_name = recall_overlap_outlier_df.columns[i]
            recall_results_mean_std_subset = recall_overlap_outlier_df.groupby(latent_prop_list).agg(
                mean=(column_name, 'mean'),
                std=(column_name, 'std'),
            ).reset_index().round(2)

            recall_results_mean_std_subset.rename(columns={'mean': '{}_mean'.format(recall_overlap_outlier_df.columns[i]), 'std': '{}_std'.format(recall_overlap_outlier_df.columns[i])}, inplace=True)
            print('subset columns: {}'.format(recall_results_mean_std_subset.columns))
            if i == 3:
                recall_results_mean_std = recall_results_mean_std_subset
            else:
                print('total_columns: {}'.format(recall_results_mean_std.columns))
                recall_results_mean_std = recall_results_mean_std.merge(recall_results_mean_std_subset, how='inner', on=['latent_learnt_method', 'outlier_prop'])
        recall_results_mean_std.to_excel('./output/VanillaCVAE_VanillaCVAELOE_VanillaCVAEOE_erosion_muscle_cut_mean_std.xlsx', header=True, index=False)
        column_name_list = list(recall_results_mean_std.columns)
        column_name_testset = [column_name for column_name in column_name_list if 'test' in column_name]
        column_name_testset = ['latent_learnt_method', 'outlier_prop'] + column_name_testset
        subset_df = recall_results_mean_std.loc[:, column_name_testset]
        print(subset_df.to_latex(index=False))

if __name__ == '__main__':
    config = optionFlags()
    main(config)
