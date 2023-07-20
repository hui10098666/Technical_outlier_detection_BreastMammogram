import os
import itertools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import argparse
import math

def get_results_dataframe(data_dir, method, MC, config_number, data_type, result_dir):
    data_df_path = data_dir + 'imageinfo_clinical_{}_MC{}.xlsx'.format(data_type, MC)
    if os.path.isfile(data_df_path):
        data_df = pd.read_excel(data_df_path)

    if method == 'threshold_erosion':
        results_path1 = result_dir + 'find_potential_outliers_by_threshold_erosion_train_taskid{}.xlsx'.format(config_number)
        results_path2 = result_dir + 'find_potential_outliers_by_threshold_erosion_test_taskid{}.xlsx'.format(config_number)
    elif method == 'muscle_cut':
        results_path1 = result_dir + 'find_potential_outliers_by_muscle_cut_train_taskid{}.xlsx'.format(config_number)
        results_path2 = result_dir + 'find_potential_outliers_by_muscle_cut_test_taskid{}.xlsx'.format(config_number)

    reference_df1 = pd.read_excel(results_path1)
    reference_df2 = pd.read_excel(results_path2)
    reference_df = pd.concat([reference_df1, reference_df2], axis=0)

    images_sha256_selected = list(data_df['image_data_sha256'])
    result_dataframe = reference_df[reference_df.image_data_sha256.isin(images_sha256_selected)].copy()

    return result_dataframe

def load_result(data_dir, result_dir, hyperparameter_config, method, MCs):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    configs_dataframe = pd.DataFrame(hyperparameter_experiments)
    configs_dataframe['config_number'] = list(range(configs_dataframe.shape[0]))
    results_list = []

    for MC in range(MCs):
        print('MC: {}'.format(MC))
        for config_number in range(len(hyperparameter_experiments)):
            print('config_number: {}'.format(config_number))
            result_dict_oneMC_oneConfig = {'method': method, 'MC': MC, 'config_number': config_number}

            for data_type in ['train','test']:
                data_df_path = data_dir + 'imageinfo_clinical_{}_MC{}.xlsx'.format(data_type, MC)
                if os.path.isfile(data_df_path):
                    data_df = pd.read_excel(data_df_path)

                if method == 'threshold_erosion':
                    outlier_reference = data_df[(~data_df.inlier_outlier.isin(['inlier', 'bad_positioning - outliers'])) & data_df.pectoral_muscle_labels.eq(1)]
                elif method == 'muscle_cut':
                    outlier_reference = data_df[data_df.pectoral_muscle_labels.eq(-1)]

                result_dict_oneMC_oneConfig.update({'reference_number_{}'.format(data_type): outlier_reference.shape[0]})

                results_dataframe = get_results_dataframe(data_dir, method, MC, config_number, data_type, result_dir)
                if method == 'threshold_erosion':
                    results_dataframe_copy = results_dataframe.copy()
                    results_dataframe.sort_values(by=['image_array_sum_after_processing'], ascending=False, inplace=True)
                elif method == 'muscle_cut':
                    results_dataframe_copy = results_dataframe.copy()
                    results_dataframe = results_dataframe[(results_dataframe['line_number_in_muscle'] <= 8) & (results_dataframe['line_number_in_muscle'] > 0)]
                    results_dataframe.sort_values(by=['line_number_in_muscle'], ascending=False, inplace=True)

                for prop in [1, 2, 5]:
                    number = math.ceil(results_dataframe_copy.shape[0]*prop/100)

                    if results_dataframe.shape[0]<number:
                        result_dict_oneMC_oneConfig.update({'detected_in_{}percent_{}'.format(prop, data_type): float("nan")})
                        result_dict_oneMC_oneConfig.update({'detected_in_{}percent_{}_percent'.format(prop, data_type): float("nan")})
                    else:
                        results_dataframe_subset = results_dataframe.iloc[0:number, :].copy()
                        detected_outliers_df = results_dataframe_subset.merge(outlier_reference, how='inner', on='image_data_sha256')
                        result_dict_oneMC_oneConfig.update({'detected_in_{}percent_{}'.format(prop, data_type): detected_outliers_df.shape[0]})
                        result_dict_oneMC_oneConfig.update({'detected_in_{}percent_{}_percent'.format(prop, data_type): detected_outliers_df.shape[0]/outlier_reference.shape[0]*100})

            results_list.append(result_dict_oneMC_oneConfig)

    results_dataframe=pd.DataFrame(results_list)
    configs_results = configs_dataframe.merge(results_dataframe, how='left', on='config_number')
    configs_results.drop(['config_number'], axis=1, inplace=True)

    return configs_results

def main( ):

    parser = argparse.ArgumentParser(description='resultsummary')

    parser.add_argument('--MCs', type=int, default=2,
                        help='number of Monte Carlos')
    parser.add_argument('--method', type=str, default='threshold_erosion',
                        help='image processing method to find potential outliers')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)
    args = parser.parse_args()

    data_dir = './data/{}/'.format(args.method)
    result_dir = './output/{}/'.format(args.method)

    if args.method == 'threshold_erosion':
        hyperparameter_config = {
            'erosion_threshold': [200, 220, 240, 180, 160],
            'erosion_kernel_size': [5, 10],
            'erosion_iteration_number': [5, 10]
        }
    elif args.method == 'muscle_cut':
        hyperparameter_config = {
            'resize_width': [256],
            'resize_height': [256],
            'muscle_cut_distance_lower': [5, 10], #[5, 20]
            'muscle_cut_distance_upper': [256], #182
            'muscle_cut_angle_lower': [10],
            'muscle_cut_angle_upper': [70],
            'muscle_line_canny_hthresh_lower': [160, 170],
            'muscle_line_canny_hthresh_upper': [180, 220],
            'muscle_line_hough_thresh': [40, 50, 60]
        }

    configs_results = load_result(data_dir, result_dir, hyperparameter_config, args.method, args.MCs)
    configs_results.to_excel(result_dir + 'configs_results.xlsx', header=True, index=False)

    config_keys_list = []
    for key in hyperparameter_config.keys():
        config_keys_list.append(key)

    configs_results_mean_std = configs_results.groupby(config_keys_list).agg(
        detected_in_1percent_train_mean=('detected_in_1percent_train_percent', 'mean'),
        detected_in_1percent_train_std=('detected_in_1percent_train_percent', 'std'),
        detected_in_2percent_train_mean=('detected_in_2percent_train_percent', 'mean'),
        detected_in_2percent_train_std=('detected_in_2percent_train_percent', 'std'),
        detected_in_5percent_train_mean=('detected_in_5percent_train_percent', 'mean'),
        detected_in_5percent_train_std=('detected_in_5percent_train_percent', 'std'),
        detected_in_1percent_test_mean=('detected_in_1percent_test_percent', 'mean'),
        detected_in_1percent_test_std=('detected_in_1percent_test_percent', 'std'),
        detected_in_2percent_test_mean=('detected_in_2percent_test_percent', 'mean'),
        detected_in_2percent_test_std=('detected_in_2percent_test_percent', 'std'),
        detected_in_5percent_test_mean=('detected_in_5percent_test_percent', 'mean'),
        detected_in_5percent_test_std=('detected_in_5percent_test_percent', 'std')
    ).reset_index().round(1)

    configs_results_mean_std.to_excel(result_dir + 'configs_results_mean_std.xlsx', header=True, index=False)

if __name__ == '__main__':
    main( )
