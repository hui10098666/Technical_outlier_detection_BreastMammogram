import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from source.image_preprocessing import image_preprocessing
from source.helpful_funs import line_number_count_in_muscle
import torchvision.transforms as transforms
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

class pixel_sum_after_erosion( ):

    def __init__(self, csv_file, root_dir, config):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df = csv_file
        self.root_dir = root_dir
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_laterality = self.df.iloc[idx].image_laterality
        use_crop = True
        img_path = self.root_dir + self.df.iloc[idx].image_relative_filepath
        img = image_preprocessing(img_path, use_crop, image_laterality)

        ret, th = cv.threshold(np.array(img), self.config.erosion_threshold, 255, cv.THRESH_BINARY)
        kernel = np.ones((self.config.erosion_kernel_size, self.config.erosion_kernel_size), np.uint8)
        img_erosion = cv.erode(th, kernel, iterations=self.config.erosion_iteration_number)
        #img_dilation = cv.dilate(img_erosion, kernel, iterations=5)

        img_array_sum = np.sum(img_erosion)

        return idx, img_array_sum

class muscle_cut_with_canny( ):

    def __init__(self, csv_file, root_dir, transform, config):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_view_position = self.df.iloc[idx].image_view_position
        if image_view_position == 'MLO':
            image_laterality = self.df.iloc[idx].image_laterality
            use_crop = True
            img_path = self.root_dir + self.df.iloc[idx].image_relative_filepath
            img = image_preprocessing(img_path, use_crop, image_laterality)

            img_resize = self.transform(img)
            sigma = 5
            lines_number = line_number_count_in_muscle(img_resize, sigma,
                                        config.muscle_cut_distance_lower, config.muscle_cut_distance_upper,
                                        config.muscle_cut_angle_lower, config.muscle_cut_angle_upper,
                                        config.muscle_line_canny_hthresh_lower, config.muscle_line_canny_hthresh_upper,
                                        config.muscle_line_hough_thresh)
        else:
            lines_number = 0
        return idx, lines_number

def optionFlags():
    parser = argparse.ArgumentParser(description='breast image')

    parser.add_argument('--taskid', type=int, default=0,
                        help='taskid from sbatch')

    parser.add_argument("--seed", type=int, default=42,
                        help="Defines the seed (default is 42)")

    parser.add_argument('--dataset_name', type=str, default='BRAIX',
                        help='the name of the dataset')

    parser.add_argument('--find_outliers_image_processing_way', type=str, default='threshold_erosion',
                        help='the way of image processing to find outliers')

    parser.add_argument('--generate_train_test_split', action='store_true', default=False,
                        help='whether to generate train test split of dataset')

    parser.add_argument('--train_size', type=float, default=0.6,
                        help='train size ratio of the dataset')

    parser.add_argument('--erosion_threshold', type=float, default=250,
                        help='the threshold to binarize breast mammogram')

    parser.add_argument('--erosion_kernel_size', type=int, default=5,
                        help='the size of the kernel to erode binary breast mammogram')

    parser.add_argument('--erosion_iteration_number', type=int, default=5,
                        help='the erosion number')

    parser.add_argument('--resize_width', type=int, default=256,
                        help='resize width for pectoral muscle cut')

    parser.add_argument('--resize_height', type=int, default=256,
                        help='resize hieight for pectoral muscle cut')

    parser.add_argument('--muscle_cut_canny_hthresh_lower', type=float, default=0.2,
                        help='the lower percentile for the hysteresis thresholding during the muscle cut')

    parser.add_argument('--muscle_cut_canny_hthresh_upper', type=float, default=0.5,
                        help='the upper percentile for the hysteresis thresholding during the muscle cut')

    parser.add_argument('--muscle_cut_distance_lower', type=float, default=10,
                        help='lower distance threshold for muscle cut')

    parser.add_argument('--muscle_cut_distance_upper', type=float, default=182,
                        help='upper distance threshold for muscle cut')

    parser.add_argument('--muscle_cut_angle_lower', type=float, default=10,
                        help='the lower angle threshold for muscle cut')

    parser.add_argument('--muscle_cut_angle_upper', type=float, default=70,
                        help='the upper angle threshold for muscle cut')

    parser.add_argument('--muscle_line_canny_hthresh_lower', type=float, default=0.55,
                        help='the lower percentile for the hysteresis thresholding for muscle line')

    parser.add_argument('--muscle_line_canny_hthresh_upper', type=float, default=0.7,
                        help='the upper percentile for the hysteresis thresholding for muscle line')

    parser.add_argument('--muscle_line_hough_thresh', type=int, default=50,
                        help='the pixel number threshold for muscle line count')

    # general usage
    parser.add_argument('--MC', type=int, default=0,
                        help='number of MC')

    parser.add_argument('--MCs', type=int, default=20,
                        help='number of repetitions')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers to load data')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    config = parser.parse_args()

    return config

def main(config):

    if config.find_outliers_image_processing_way == 'threshold_erosion':
        if not os.path.exists('./data/threshold_erosion/'):
            os.makedirs('./data/threshold_erosion/')
        data_dir = './data/threshold_erosion/'
    elif config.find_outliers_image_processing_way == 'muscle_cut':
        if not os.path.exists('./data/muscle_cut/'):
            os.makedirs('./data/muscle_cut/')
        data_dir = './data/muscle_cut/'

    if config.generate_train_test_split:

        imageinfo_clinical = pd.read_excel('./data/Hui_BRAIX_dataset_info.xlsx')
        subset = imageinfo_clinical[~imageinfo_clinical["image_manufacturer_algorithm"].str.contains("Implant|Imp")]
        subset = subset[(subset.image_manufacturer_algorithm.eq('None')) | (subset["image_manufacturer_algorithm"].str.contains("F1|Flavour1|Flavor1|HC_Standard_OV2"))]
        imageinfo_clinical = subset[~subset.image_breast_implant_present.eq(True)]

        if config.find_outliers_image_processing_way == 'muscle_cut':
            imageinfo_clinical_copy = imageinfo_clinical.copy()
            imageinfo_clinical = imageinfo_clinical_copy[imageinfo_clinical_copy.image_view_position.eq('MLO')]

        true_outliers_dataset_total = pd.read_excel('./data/imageinfo_clinical_PotentialOutliers_for_radiologist.xlsx')
        true_outliers_dataset = true_outliers_dataset_total[true_outliers_dataset_total.inlier_outlier_labels.eq(-1)].copy()
        if config.find_outliers_image_processing_way == 'muscle_cut':
            true_outliers_dataset = true_outliers_dataset_total[(true_outliers_dataset_total.inlier_outlier_labels.eq(-1)) & (true_outliers_dataset_total.image_view_position.eq('MLO'))].copy()

        true_outliers_dataset_subset = true_outliers_dataset.loc[:, ['image_data_sha256', 'inlier_outlier', 'inlier_outlier_labels', 'pectoral_muscle', 'pectoral_muscle_labels']].copy()

        imageinfo_clinical_outlier = imageinfo_clinical.merge(true_outliers_dataset_subset, how='inner', on='image_data_sha256')
        imageinfo_clinical_inlier = imageinfo_clinical[~imageinfo_clinical.image_data_sha256.isin(list(true_outliers_dataset_subset['image_data_sha256']))].copy()
        imageinfo_clinical_inlier['inlier_outlier'] = 'inlier'
        imageinfo_clinical_inlier['inlier_outlier_labels'] = 1
        imageinfo_clinical_inlier['pectoral_muscle'] = 'pectoral_muscle - no outliers'
        imageinfo_clinical_inlier['pectoral_muscle_labels'] = 1
        imageinfo_clinical_total = pd.concat([imageinfo_clinical_outlier, imageinfo_clinical_inlier], axis=0)

        # generate a random seed to split training and testing dataset
        np.random.seed(1011)
        desired_seeds = np.random.randint(0, 2 ** 32, size=(1, config.MCs), dtype=np.uint32)

        for MC in range(config.MCs):
            seed_one_MC =  int(desired_seeds[0, MC])
            imageinfo_clinical_train, imageinfo_clinical_test = sklearn.model_selection.train_test_split(imageinfo_clinical_total, train_size = config.train_size, random_state=seed_one_MC)
            imageinfo_clinical_train.to_excel(data_dir + 'imageinfo_clinical_train_MC{}.xlsx'.format(MC), header=True, index=False)
            imageinfo_clinical_test.to_excel(data_dir + 'imageinfo_clinical_test_MC{}.xlsx'.format(MC), header=True, index=False)
    else:
        imageinfo_clinical_train = pd.read_excel(data_dir + 'imageinfo_clinical_train_MC0.xlsx')
        imageinfo_clinical_test = pd.read_excel(data_dir + 'imageinfo_clinical_test_MC0.xlsx')

        root_dir = '/mnt/beegfs/mccarthy/scratch/projects/braix/'

        for dataset_type in ['train', 'test']:
            imageinfo_clinical_dataset = imageinfo_clinical_train if dataset_type == 'train' else imageinfo_clinical_test
            selected_outliers = []
            if config.find_outliers_image_processing_way == 'threshold_erosion':
                pixel_sum = pixel_sum_after_erosion(imageinfo_clinical_dataset, root_dir, config)
                for i in range(imageinfo_clinical_dataset.shape[0]):
                    RowIndex_PixelSum = pixel_sum.__getitem__(i)
                    selected_outliers += [RowIndex_PixelSum]

                selected_outliers_dataframe = pd.DataFrame(selected_outliers, columns=['row_index', 'image_array_sum_after_processing'])

            elif config.find_outliers_image_processing_way == 'muscle_cut':
                transform = transforms.Compose([
                    transforms.Resize((config.resize_width, config.resize_height)),
                ])
                muscle_cut = muscle_cut_with_canny(imageinfo_clinical_dataset, root_dir, transform, config)
                for i in range(imageinfo_clinical_dataset.shape[0]):
                    RowIndex_LineNumber = muscle_cut.__getitem__(i)
                    selected_outliers += [RowIndex_LineNumber]

                selected_outliers_dataframe = pd.DataFrame(selected_outliers, columns=['row_index', 'line_number_in_muscle'])

            imageinfo_clinical_dataset['row_index'] = list(range(imageinfo_clinical_dataset.shape[0]))
            imageinfo_clinical_result = imageinfo_clinical_dataset.merge(selected_outliers_dataframe, how='inner', on='row_index')
            imageinfo_clinical_result.drop('row_index', axis=1, inplace=True)

        if not os.path.exists('./output/PotentialOutliers/{}/'.format(config.find_outliers_image_processing_way)):
            os.makedirs('./output/PotentialOutliers/{}/'.format(config.find_outliers_image_processing_way))
        config.output_dir = './output/PotentialOutliers/{}/'.format(config.find_outliers_image_processing_way)

        if config.find_outliers_image_processing_way == 'threshold_erosion':
            imageinfo_clinical_result.to_excel(config.output_dir + 'find_potential_outliers_by_threshold_erosion_{}_taskid{}.xlsx'.format(dataset_type, config.taskid), header=True, index=False)
        elif config.find_outliers_image_processing_way == 'muscle_cut':
            imageinfo_clinical_result.to_excel(config.output_dir + 'find_potential_outliers_by_muscle_cut_{}_taskid{}.xlsx'.format(dataset_type, config.taskid), header=True, index=False)

if __name__ == '__main__':
    config = optionFlags()
    main(config)
