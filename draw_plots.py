import os
import pickle
import glob
import argparse
import pandas as pd
import math
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools


'''
def imshow(inp, save_path):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(20,20))
    plt.imshow(inp, vmin=0., vmax=1.)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig(save_path)
    
file_paths = configfiles = glob.glob('./output/wandb/ResNetCVAE/**/*.pkl', recursive=True)
for file_path in file_paths:
    pickle_file = pickle.load(open(file_path, 'rb'))
    if len(pickle_file['data_idx'])==64:
        #only get 16 images, make_util add padding=2 to images: 512*4 + 2*5
        InputImage = pickle_file['InputImage'][:,0:2058,:]
        ReconstImage = pickle_file['ReconstructedImage'][:,0:2058,:]
    else:
        InputImage = pickle_file['InputImage']
        ReconstImage = pickle_file['ReconstructedImage']

    if 'batch0' in file_path:
        input_save_path = os.path.dirname(file_path) + '//InputImage_batch0.png'
        reconst_save_path = os.path.dirname(file_path) + '//ReconstImage_batch0.png'
    elif 'batch1' in file_path:
        input_save_path = os.path.dirname(file_path) + '//InputImage_batch1.png'
        reconst_save_path = os.path.dirname(file_path) + '//ReconstImage_batch1.png'

    imshow(InputImage, input_save_path)
    imshow(ReconstImage, reconst_save_path)

file_paths = glob.glob('./output/simulation/ResNetCVAE/PotentialOutliers/*.pkl', recursive=True)
for file_path in file_paths:
    pickle_file = pickle.load(open(file_path, 'rb'))
    InputImage = pickle_file['InputImage']
    input_save_path = file_path.replace('.pkl','.png')
    imshow(InputImage, input_save_path)

#find all outliers
# the outlier_index_among_100 is from 1 to 100, so we need to minus 1.
PotentialOutliers_index = pd.read_excel('./output/PotentialOutliers/PotentialOutliers_index.xlsx')
PotentialOutliers_index['index_in_imageinfo_clinical_PotentialOutliers_dataset'] = 0
for i in range(PotentialOutliers_index.shape[0]):
    pickle_path = './output/PotentialOutliers/' + PotentialOutliers_index.iloc[i].pkl_path + '.pkl'
    pickle_file = pickle.load(open(pickle_path, 'rb'))
    original_index = pickle_file['original_index']
    PotentialOutliers_index.at[PotentialOutliers_index.index[i], 'index_in_imageinfo_clinical_PotentialOutliers_dataset'] = original_index[int(PotentialOutliers_index.iloc[i].outlier_index_among_100) - 1]

PotentialOutliers_index.to_excel('./output/PotentialOutliers/PotentialOutliers_index.xlsx', header=True, index=False)


#preprocess the true outlier folder from peter
def generate_true_inlier_outlier_labels(row):
    if row['inlier_outlier'] in ['dense_tissue_behind_breast - no outliers', 'extra_fat_skin - not outliers', 'Should not be outliers']:
        return 1
    else:
        return -1
        
def generate_pectoral_muscle_labels(row):
    if row['pectoral_muscle'] in ['pectoral_muscle - outliers']:
        return -1
    else:
        return 1
    
imageinfo_clinical_outliers = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
dirs_list = ['bad_positioning - outliers', 'bad_radiography - outliers', 'dense_tissue_behind_breast - no outliers', 'extra_fat_skin - not outliers', 'implants - outliers', 'medical_devices_line - outliers', 'medical_devices_eclipse - outliers', 'Should not be outliers', 'Unusual lesions-Calcifications - outliers', 'wrong_algorithm - outliers']
all_PotentialOutliers_path_list = list(imageinfo_clinical_outliers['image_relative_filepath'])
imageinfo_clinical_outliers['inlier_outlier'] = 'None'
for dir in dirs_list:
    file_paths = glob.glob('./data/true_outlier_images/{}/*.png'.format(dir), recursive=True)
    for i in range(len(file_paths)):
        image_name = file_paths[i].split('\\')[1]
        if image_name[0].isdigit():
            image_name = image_name.split('_', 1)[1]
        print(image_name)
        indices = [i for i, elem in enumerate(all_PotentialOutliers_path_list) if image_name in elem]
        print(indices)
        if len(indices) == 0 or len(indices) > 1:
            print('There is some error!')
        else:
            imageinfo_clinical_outliers.at[imageinfo_clinical_outliers.index[indices[0]], 'inlier_outlier'] = dir

col = imageinfo_clinical_outliers.apply(lambda row: generate_true_inlier_outlier_labels(row), axis=1)  # get column data with an index
imageinfo_clinical_outliers = imageinfo_clinical_outliers.assign(inlier_outlier_labels=col.values)
imageinfo_clinical_outliers.to_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx', header=True, index=False)
'''

def generate_feature_method(row):
    if row['feature'] in ['negReconstLoss', 'negKLDLoss', 'ELBO', 'ensemble1', 'ensemble2']:
        return row['feature']
    else:
        return row['feature'] + '_' + row['method']

def select_outlier_scores(dataframe, config):

    dataframe_largest15 = dataframe.sort_values('mean', ascending=False).groupby(['outlier_prop', 'outlier_type', 'metric', 'model'], sort=False).head(10)
    dataframe_largest15_FeatureMethod = dataframe_largest15.groupby(['feature', 'method'])['outlier_prop'].agg('count').reset_index()
    dataframe_largest15_FeatureMethod.rename(columns={'outlier_prop': 'count'}, inplace=True)
    col = dataframe_largest15_FeatureMethod.apply(lambda row: generate_feature_method(row), axis=1)
    dataframe_largest15_FeatureMethod = dataframe_largest15_FeatureMethod.assign(feature_method=col.values)

    Selected_FeatureMethod_list = list(dataframe_largest15_FeatureMethod[dataframe_largest15_FeatureMethod['count'] == len(config.outlier_props)].loc[:, 'feature_method'])
    Selected_FeatureMethod_list2 = ['latent_LOF', 'latent_OCSVM',  'negKLDLoss', 'negKLDLoss_latent_OCSVM',
                                    'negReconstLoss_latent_LOF', 'negKLDLoss_UMAP_OCSVM', 'ELBO_latent_LOF', 'ELBO',
                                    'negReconstLoss', 'negKLDLoss_latent_LOF']
    Selected_FeatureMethod_list = [k for k in Selected_FeatureMethod_list if k in Selected_FeatureMethod_list2]

    if len(Selected_FeatureMethod_list) == 0:
        print('no feature method satisfied the requirement')

    return Selected_FeatureMethod_list

def draw_line_plot(dataframe, title_variable, full_FeatureMethod_list, color_list, config):

    if config.each_row == 'metric':
        subplots_row_list = config.metric_list,
    else:
        subplots_row_list = config.outlier_types
    subplot_column_list = ['VanillaCVAE', 'ResNetCVAE']

    col = dataframe.apply(lambda row: generate_feature_method(row), axis=1)
    dataframe = dataframe.assign(feature_method=col.values)

    dataframe = dataframe[dataframe.outlier_prop.isin(config.outlier_props)]

    if subplot_column_list[0] in config.model_list:
        subplot_title_tuple = ('VanillaCVAE', 'ResNetCVAE')

    fig = make_subplots(rows=len(subplots_row_list), cols=len(subplot_column_list), shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=subplot_title_tuple, vertical_spacing=0.02, horizontal_spacing=0.02)

    for i, j in list(itertools.product(list(range(len(subplots_row_list))),list(range(len(subplot_column_list))))):

        if subplots_row_list[i] in config.metric_list and subplot_column_list[j] in config.model_list:
            dataframe_copy = dataframe.copy()
            dataframe_copy = dataframe_copy[dataframe_copy.outlier_type.eq(title_variable)]
            dataframe_subset = dataframe_copy.loc[(dataframe_copy['metric'] == subplots_row_list[i]) & (dataframe_copy['model'] == subplot_column_list[j])]
        elif subplots_row_list[i] in config.outlier_types and subplot_column_list[j] in config.model_list:
            dataframe_copy = dataframe.copy()
            dataframe_copy = dataframe_copy[dataframe_copy.outlier_type.eq(subplots_row_list[i])]
            dataframe_subset = dataframe_copy.loc[(dataframe_copy['metric'] == 'auprc') & (dataframe_copy['model'] == subplot_column_list[j])]

        #Selected_FeatureMethod_list = select_outlier_scores(dataframe=dataframe_subset, config=config)

        if j==0:
            Selected_FeatureMethod_list = ['negReconstLoss', 'negKLDLoss', 'latent_OCSVM', 'ensemble3_average']
        else:
            Selected_FeatureMethod_list = ['negReconstLoss', 'negKLDLoss', 'latent_OCSVM', 'ensemble4_average']

        #Selected_FeatureMethod_list = ['ensemble1_average', 'ensemble2_average','ensemble3_average', 'ensemble4_average'] #

        if len(Selected_FeatureMethod_list) == 0:
            x_value = config.outlier_props
            y_value = [0] * len(config.outlier_props)
            std_value = [0] * len(config.outlier_props)
            color_value = 'white'

            fig.add_trace(
                go.Scatter(
                    mode='markers+lines',
                    x=x_value,
                    y=y_value,
                    error_y=dict(
                        type='data',
                        array=std_value,
                        color=color_value,
                        visible=True),
                    marker=dict(
                        color=color_value,
                        size=10
                    ),
                    line=dict(
                        color=color_value
                    ),
                    showlegend=False
                ),
                row=i+1, col=j+1)
        else:
            for one_feature_method in Selected_FeatureMethod_list:
                dataframe_subset_one_feature_method = dataframe_subset[dataframe_subset.feature_method.eq(one_feature_method)]
                dataframe_subset_one_feature_method_copy = dataframe_subset_one_feature_method.copy()
                feature_method_index = full_FeatureMethod_list.index(one_feature_method)

                dataframe_subset_one_feature_method_copy = dataframe_subset_one_feature_method_copy.sort_values('outlier_prop', ascending=True)

                print('{}, {}, std: {}'.format(one_feature_method, subplots_row_list[i], list(dataframe_subset_one_feature_method_copy['std'])))

                x_value = dataframe_subset_one_feature_method_copy['outlier_prop']
                y_value = dataframe_subset_one_feature_method_copy['mean']
                std_value = dataframe_subset_one_feature_method_copy['std']
                color_value = color_list[feature_method_index]

                fig.add_trace(
                    go.Scatter(
                        mode='markers+lines',
                        x=x_value,
                        y=y_value,
                        error_y=dict(
                            type='data',
                            array=std_value,
                            color=color_value,
                            visible=True),
                        marker=dict(
                            color=color_value,
                            size=10
                        ),
                        line=dict(
                            color=color_value
                        ),
                        name=one_feature_method,
                        showlegend=True
                    ),
                    row=i+1, col=j+1)

    names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

    fig.update_annotations(font_size=22)

    if config.each_row == 'metric':
        supertitle = 'Outlier type: {}'.format(title_variable)
    else:
        supertitle = 'auprc'

    fig.update_layout(
        width=1200,
        height=1200,
        margin=dict(
              l=80,
              r=90,
              b=70,
              t=70,
              pad=1
        ),
        font=dict(color='black', family='Times New Roman', size=25),
        title={
            'text': supertitle,
            'font': {
                'size': 25
            },
            'y': 0.99,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title_text='Outlier scores',
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=20,
                color='black'
            )
        ),
       plot_bgcolor ='rgb(255, 255, 255)',
       paper_bgcolor = 'rgb(255, 255, 255)'
    )

    if config.each_row == 'metric':
        for i in range(len(subplots_row_list)):
            if subplots_row_list[i] in ['auroc', 'auprc']:
                fig.update_yaxes(title_text="{}".format(subplots_row_list[i]), title_font_size=22, row = i+1, col = 1)
            elif subplots_row_list[i] in ['precision', 'recall']:
                fig.update_yaxes(title_text="max {}".format(subplots_row_list[i]), title_font_size=22, row = i+1, col = 1)
            elif subplots_row_list[i] == 'F1_score':
                fig.update_yaxes(title_text="max F1 score", title_font_size=22, row = i + 1, col = 1)
    else:
        #correspond to 'all', 'flip_half_breast', 'straight_medical_device', 'ellipse_medical_device', 'wrong_algorithm'
        corrected_outlier_names = ['all', 'bad_positioning', 'medical_device_line', 'medical_device_eclipse', 'wrong_algorithm']
        for i in range(len(subplots_row_list)):
            fig.update_yaxes(title_text="{}".format(corrected_outlier_names[i]), title_font_size=22, row = i+1, col = 1)

    for j in range(len(subplot_column_list)):
        fig.update_xaxes(title_text="proportion of outliers", title_font_size=22, row=len(subplots_row_list), col=j+1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, zeroline=False)
    fig.update_xaxes(tickvals=[0.01, 0.05, 0.1], ticks="outside", tickwidth=1,tickcolor='black',
                     ticklen=6, tickfont=dict(family='Times New Roman', size=18))
    fig.update_yaxes(range=[0,1.1], nticks=12,  ticks="outside", tickwidth=1, tickcolor='black',
                     ticklen=6, tickfont=dict(family='Times New Roman', size=18))

    if title_variable in config.outlier_types:
        fig.write_image(config.save_dir + 'AllMetrics_AllModels_OutlierType_{}.png'.format(title_variable))
    else:
        fig.write_image(config.save_dir + 'AUPRC_AllModels_OneOutlierType.png')

def draw_box_plot(dataframe, config, title_variable):

    col = dataframe.apply(lambda row: generate_feature_method(row), axis=1)
    dataframe = dataframe.assign(feature_method=col.values)


    feature_method_categories = ['negReconstLoss', 'negKLDLoss', 'ELBO', 'latent_IF', 'latent_LOF', 'latent_OCSVM',
                                 'negReconstLoss_latent_IF', 'negReconstLoss_latent_LOF',
                                 'negReconstLoss_latent_OCSVM', 'negKLDLoss_latent_IF', 'negKLDLoss_latent_LOF',
                                 'negKLDLoss_latent_OCSVM', 'ELBO_latent_IF', 'ELBO_latent_LOF', 'ELBO_latent_OCSVM']

    if config.ensemble:
        feature_method_categories += ['ensemble1', 'ensemble2']

    dataframe = (dataframe[dataframe.feature_method.isin(feature_method_categories)]).copy()

    subplot_title_list = config.outlier_types.copy()

    if 'bad_positioning - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('bad_positioning - outliers')] = 'improper placement'
    if 'bad_radiography - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('bad_radiography - outliers')] = 'improper radiography'
    if 'implants - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('implants - outliers')] = 'implant'
    if 'Unusual lesions-Calcifications - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('Unusual lesions-Calcifications - outliers')] = 'atypical lesion / calcification'
    if 'wrong_algorithm - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('wrong_algorithm - outliers')] = 'incorrect exposure parameter'
    if 'medical_devices_eclipse - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('medical_devices_eclipse - outliers')] = 'pacemaker'
    if 'medical_devices_line - outliers' in config.outlier_types:
        subplot_title_list[subplot_title_list.index('medical_devices_line - outliers')] = 'cardiac loop recorder'

    if config.each_row == 'metric':
        fig = make_subplots(rows=len(config.metric_list), cols=1, shared_xaxes=True, vertical_spacing=0.02)
        row_list = config.metric_list
    elif config.each_row == 'outlier_type':
        fig = make_subplots(rows=len(config.outlier_types), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=['Outlier type: {}'.format(k) for k in subplot_title_list])
        row_list = config.outlier_types

    for i in range(len(row_list)):
        if config.each_row == 'metric':
            metric = config.metric_list[i]
            outlier_type = title_variable
        elif config.each_row == 'outlier_type':
            metric = 'auprc'
            outlier_type = config.outlier_types[i]

        dataframe_subset_one_metric_one_outlier = dataframe.loc[(dataframe['metric'] == metric) & (dataframe['outlier_type'] == outlier_type)]
        for one_model in config.model_list:
            for data_type in ['train','test']:
                dataframe_subset_OneModel_TrainTest = (dataframe_subset_one_metric_one_outlier[(dataframe_subset_one_metric_one_outlier.model.eq(one_model)) &
                dataframe_subset_one_metric_one_outlier.train_test.eq(data_type)]).copy()

                dataframe_subset_OneModel_TrainTest.feature_method = pd.Categorical(dataframe_subset_OneModel_TrainTest.feature_method, categories=feature_method_categories)
                dataframe_subset_OneModel_TrainTest = dataframe_subset_OneModel_TrainTest.sort_values('feature_method')

                if dataframe_subset_OneModel_TrainTest.shape[0] > 0:

                    feature_method_list = list(dataframe_subset_OneModel_TrainTest['feature_method'])
                    print('len: {}'.format(len(feature_method_list)))
                    print(feature_method_list)
                    if one_model == 'VanillaCVAE' and data_type=='train':
                        color = 'lightpink'
                    elif one_model == 'VanillaCVAE' and data_type=='test':
                        color = 'deeppink'
                    elif one_model == 'ResNetCVAE' and data_type=='train':
                        color = 'lightgray'
                    elif one_model == 'ResNetCVAE' and data_type=='test':
                        color = 'darkgray'

                    if one_model == 'VanillaCVAE' and data_type=='train':
                        legendrank = 1
                    elif one_model == 'VanillaCVAE' and data_type=='test':
                        legendrank = 2
                    elif one_model == 'ResNetCVAE' and data_type=='train':
                        legendrank = 3
                    elif one_model == 'ResNetCVAE' and data_type=='test':
                        legendrank = 4

                    fig.add_trace(
                        go.Bar(
                            x=dataframe_subset_OneModel_TrainTest['feature_method'],
                            y=dataframe_subset_OneModel_TrainTest['mean'],
                            error_y=dict(type='data', array=dataframe_subset_OneModel_TrainTest['std']),
                            name='{}, {}'.format(one_model, data_type),
                            marker_color = color,
                            showlegend=True,
                            legendrank=legendrank
                        ),
                        row=i + 1, col=1)

        if config.each_row == 'metric':
            if config.metric_list[i] == 'auroc':
                fig.update_yaxes(title_text="{}".format(config.metric_list[i]), title_font_size=25, range=[0,1], row=i + 1, col=1)
            elif config.metric_list[i] == 'auprc':
                fig.update_yaxes(title_text="{}".format(config.metric_list[i]), title_font_size=25, range=[0, 0.4], row=i + 1, col=1)
            elif config.metric_list[i] in ['precision', 'recall']:
                fig.update_yaxes(title_text="max {}".format(config.metric_list[i]), title_font_size=25, range=[0,1], row=i + 1, col=1)
            elif config.metric_list[i] == 'F1_score':
                fig.update_yaxes(title_text="max F1 score", title_font_size=25, range=[0,0.4], row=i + 1, col=1)
        elif config.each_row == 'outlier_type':
            if outlier_type in ['bad_positioning - outliers', 'wrong_algorithm - outliers']:
                fig.update_yaxes(title_text="auprc", title_font_size=25, range=[0,1], row=i + 1, col=1)
            elif outlier_type in ['implants - outliers', 'medical_devices_eclipse - outliers']:
                    fig.update_yaxes(title_text="auprc", title_font_size=25, range=[0, 1], row=i + 1, col=1)
            elif outlier_type in ['bad_radiography - outliers']:
                fig.update_yaxes(title_text="auprc", title_font_size=25, range=[0, 0.25], row=i + 1, col=1)
            elif outlier_type in ['medical_devices_line - outliers']:
                fig.update_yaxes(title_text="auprc", title_font_size=25, range=[0, 0.12], row=i + 1, col=1)
            elif outlier_type in ['Unusual lesions-Calcifications - outliers']:
                fig.update_yaxes(title_text="auprc", title_font_size=25, range=[0, 0.5], row=i + 1, col=1)

    names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

    fig.update_annotations(font_size=20)

    if config.each_row == 'metric':
        supertitle = 'Outlier type: {}'.format(title_variable)
        fig.update_layout(
            font=dict(color='black', family='Times New Roman', size=25),
            title={
                'text': supertitle,
                'font': {
                    'size': 25
                },
                'y': 0.99,
                'x': 0.45,
                'xanchor': 'center',
                'yanchor': 'top'},
        )

    fig.update_layout(
        barmode='group',
        width=1200,
        height=1200,
        margin=dict(
            l=80,
            r=90,
            b=70,
            t=70,
            pad=1
        ),
        font=dict(color='black', family='Times New Roman', size=25),
        legend_title_text='Model,train/test',
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=25,
                color='black'
            )
        ),
        plot_bgcolor='rgb(255, 255, 255)',
        paper_bgcolor='rgb(255, 255, 255)'
    )

    if config.ensemble:
        fig.update_xaxes(tickangle=0, ticktext=[k + 1 for k in list(range(15))] + ['ensb1', 'ensb2'], tickfont=dict(size=20), tickvals=feature_method_list)
    else:
        fig.update_xaxes(tickangle=0, ticktext=[k+1 for k in list(range(15))], tickfont=dict(size=20), tickvals=feature_method_list)

    if config.each_row == 'metric':
        fig.update_xaxes(title_text="outlier scores", title_font_size=25, row=len(config.metric_list), col=1)
    elif config.each_row == 'outlier_type':
        fig.update_xaxes(title_text="outlier scores", title_font_size=25, row=len(config.outlier_types), col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, zeroline=False)

    fig.update_yaxes(ticks="outside", tickwidth=1, tickcolor='black',ticklen=6, tickfont=dict(family='Times New Roman', size=22))

    if config.each_row == 'metric':
        fig.write_image(config.save_dir + 'AllMetrics_AllModels_TrueOutliers_{}_boxplot.png'.format(title_variable))
    elif config.each_row == 'outlier_type':
        fig.write_image(config.save_dir + 'AUPRC_AllModels_TrueOutliers_All_boxplot.png')

def optionFlags():
    parser = argparse.ArgumentParser(description='breast image')

    parser.add_argument('--taskid', type=int, default=0,
                        help='taskid from sbatch')

    parser.add_argument("--seed", type=int, default=42,
                        help="Defines the seed (default is 42)")

    parser.add_argument('--dataset_name', type=str, default='BRAIX',
                        help='the name of the dataset')

    parser.add_argument('--simulation', action='store_true', default=False,
                        help='draw plots for simulation study or not')

    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='whether ensembled outlier score exists or not')

    parser.add_argument('--save_dir', type=str, default='./output/performance_TrueOutliers/',
                        help='the directory to save results')

    parser.add_argument('--model_list', type=list, default=['VanillaCVAE', 'ResNetCVAE'],
                        help='the list of model types')

    parser.add_argument('--outlier_props', type=list, default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                        help='the list of all outlier proportions')

    parser.add_argument('--outlier_types', type=list, default=['Unusual lesions-Calcifications - outliers', 'wrong_algorithm - outliers','bad_positioning - outliers'],
                        help='the type of outliers') #['all', 'implants - outliers', 'medical_devices_eclipse - outliers', 'medical_devices_line - outliers', 'bad_radiography - outliers', 'Unusual lesions-Calcifications - outliers', 'wrong_algorithm - outliers','bad_positioning - outliers'], ['all', 'flip_half_breast', 'straight_medical_device', 'ellipse_medical_device', 'wrong_algorithm'],['all', 'black', 'white', 'contrast', 'gauss_noise', 'same_histogram']

    parser.add_argument('--metric_list', type=list, default=['auroc','auprc','precision','recall','F1_score'],
                        help='the list of metrics')

    parser.add_argument('--plot_type', type=str, default='box',
                        help='draw line or box plot')

    parser.add_argument('--each_row', type=str, default='outlier_type',
                        help='whether each row of the plots is a different metric, or is auprc for a different outlier type ') #outlier_type

    # general usage
    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    config = parser.parse_args()

    return config

def main(config=None):

    if config.plot_type == 'box':
        config.outlier_props = [0.0]

    TotalResult_dataframe = pd.DataFrame()
    for model in ['VanillaCVAE', 'ResNetCVAE']:
        TotalResult_dataframe_OneMethod = pd.DataFrame()
        for train_test in ['train','test']:

            auroc_df = pd.read_excel(config.save_dir + '{}/auroc_auprc_avgPrecision_TotalResult_ensemble_{}.xlsx'.format(model, train_test))
            precision_df = pd.read_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_max_mean_ensemble_{}.xlsx'.format(model, train_test))

            '''
            auroc_df = pd.read_excel(config.save_dir + '{}/auroc_auprc_avgPrecision_TotalResult_{}.xlsx'.format(model, train_test))
            precision_df = pd.read_excel(config.save_dir + '{}/precision_recall_F1score_TotalResult_max_mean_{}.xlsx'.format(model, train_test))
            '''

            TotalResult_dataframe_OneMethod_TrainTest = pd.concat([auroc_df, precision_df],axis=0)

            TotalResult_dataframe_OneMethod_TrainTest['model'] = model
            TotalResult_dataframe_OneMethod_TrainTest['train_test'] = train_test

            if TotalResult_dataframe_OneMethod.shape[0]==0:
                TotalResult_dataframe_OneMethod = TotalResult_dataframe_OneMethod_TrainTest
            else:
                TotalResult_dataframe_OneMethod = pd.concat([TotalResult_dataframe_OneMethod, TotalResult_dataframe_OneMethod_TrainTest], axis=0)

        if  TotalResult_dataframe.shape[0]==0:
            TotalResult_dataframe = TotalResult_dataframe_OneMethod
        else:
            TotalResult_dataframe = pd.concat([TotalResult_dataframe, TotalResult_dataframe_OneMethod], axis=0)

    full_FeatureMethod_list = []
    features_methods = [('negReconstLoss', 'None'), ('negKLDLoss', 'None'), ('ELBO', 'None')]
    features_methods += list(itertools.product(['latent', 'negReconstLoss_latent', 'negKLDLoss_latent', 'ELBO_latent'], ['IF', 'LOF', 'OCSVM']))

    if config.ensemble:
        features_methods += [('ensemble1', 'None'), ('ensemble2', 'None')]


    for feature, method in features_methods:
        if feature in ['negReconstLoss', 'negKLDLoss', 'ELBO', 'ensemble1', 'ensemble2']:
            full_FeatureMethod_list = full_FeatureMethod_list + [feature]
        else:
            full_FeatureMethod_list = full_FeatureMethod_list + ['{}_{}'.format(feature, method)]

    color_list = [ 'purple', 'red', 'blue', 'magenta',  'orange', 'green', 'rosybrown', 'lightcyan', 'tomato',
                  'tan',  'yellow',  'lightgreen', 'gold', 'palevioletred', 'greenyellow',
                   'olive', 'pink', 'firebrick',  'navy', 'lightpink','lightskyblue', 'orchid', 'lime',
                   'chocolate', 'cyan', 'dimgray','brown', 'grey', 'black', 'lightseagreen','palegreen']

    if config.plot_type == 'line':
        if config.each_row == 'metric':
            for title_variable in config.outlier_types:
                draw_line_plot(dataframe=TotalResult_dataframe, title_variable = title_variable, full_FeatureMethod_list=full_FeatureMethod_list, color_list = color_list, config=config)
        else:
            draw_line_plot(dataframe=TotalResult_dataframe, title_variable='auprc', full_FeatureMethod_list=full_FeatureMethod_list, color_list=color_list, config=config)
    elif config.plot_type == 'box':
        if config.each_row == 'metric':
            for i in range(len(config.outlier_types)):
                draw_box_plot(dataframe=TotalResult_dataframe, config=config, title_variable=config.outlier_types[i])
        else:
            draw_box_plot(dataframe=TotalResult_dataframe, config=config, title_variable='None')

if __name__ == '__main__':
    config = optionFlags()
    main(config)