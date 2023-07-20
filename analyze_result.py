import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {

        'dataset_name': ['BRAIX'],
        'task': ['auroc_precision'], # 'recall'
        'model': ['ResNetCVAE', 'VanillaCVAE'], # 'VanillaCVAE'
        'simulated_outlier_type': ['true'],
        'train_test': ['train','test'],
        'n_boots': [20] ## --simulation --ensemble
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python result_summary.py --ensemble --taskid %s "
              "--dataset_name %s --task %s --model %s "
              "--simulated_outlier_type %s --train_test %s --n_boots %s "
              % (taskid, temp['dataset_name'], temp['task'], temp['model'],
                 temp['simulated_outlier_type'], temp['train_test'], temp['n_boots'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])