import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {
        'find_outliers_image_processing_way': ['muscle_cut'], #'threshold_erosion'
        'resize_width': [256],
        'resize_height': [256],
        'muscle_cut_distance_lower': [5, 10],
        'muscle_cut_distance_upper': [256],
        'muscle_cut_angle_lower': [10],
        'muscle_cut_angle_upper': [70],
        'muscle_line_canny_hthresh_lower': [160, 170],
        'muscle_line_canny_hthresh_upper': [180, 220],
        'muscle_line_hough_thresh': [40, 50, 60],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python find_potential_outliers_image_processing.py  --taskid %s --find_outliers_image_processing_way %s "
              "--resize_width %s --resize_height %s "
              "--muscle_cut_distance_lower %s --muscle_cut_distance_upper %s "
              "--muscle_cut_angle_lower %s --muscle_cut_angle_upper %s "
              "--muscle_line_canny_hthresh_lower %s --muscle_line_canny_hthresh_upper %s "
              "--muscle_line_hough_thresh %s "
              % (taskid, temp['find_outliers_image_processing_way'],
                 temp['resize_width'], temp['resize_height'],
                 temp['muscle_cut_distance_lower'], temp['muscle_cut_distance_upper'],
                 temp['muscle_cut_angle_lower'], temp['muscle_cut_angle_upper'],
                 temp['muscle_line_canny_hthresh_lower'], temp['muscle_line_canny_hthresh_upper'],
                 temp['muscle_line_hough_thresh'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])

'''
'erosion_threshold': [200, 220, 240, 180, 160],
'erosion_kernel_size': [5, 10],
'erosion_iteration_number': [5, 10],

'resize_width': [256],
'resize_height': [256],
'muscle_cut_distance_lower': [5, 10],
'muscle_cut_distance_upper': [256],
'muscle_cut_angle_lower': [10],
'muscle_cut_angle_upper': [70],
'muscle_line_canny_hthresh_lower': [160, 170],
'muscle_line_canny_hthresh_upper': [180, 220],
'muscle_line_hough_thresh': [40, 50, 60],
'''

'''
"--erosion_threshold %s "
"--erosion_kernel_size %s --erosion_iteration_number %s "

"--resize_width %s --resize_height %s "
"--muscle_cut_distance_lower %s --muscle_cut_distance_upper %s "
"--muscle_cut_angle_lower %s --muscle_cut_angle_upper %s "
"--muscle_line_canny_hthresh_lower %s --muscle_line_canny_hthresh_upper %s "
"--muscle_line_hough_thresh %s "  
'''

'''
temp['erosion_threshold'],
temp['erosion_kernel_size'], temp['erosion_iteration_number'],

temp['resize_width'], temp['resize_height'],
temp['muscle_cut_distance_lower'], temp['muscle_cut_distance_upper'],
temp['muscle_cut_angle_lower'], temp['muscle_cut_angle_upper'],
temp['muscle_line_canny_hthresh_lower'], temp['muscle_line_canny_hthresh_upper'],
temp['muscle_line_hough_thresh'], 
'''