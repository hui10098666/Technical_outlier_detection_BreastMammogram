import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {

        'dataset_name': ['BRAIX'],
        'model': ['ResNetCVAE'],
        'resize_height': [256],
        'resize_width': [256],
        'image_channels': [1],
        'first_out_channels': [32, 16, 8],
        'latent_dim': [128, 256, 512, 1024],
        'train_set_ratio': [0.6], #
        'valid_set_ratio': [0.1],
        'optimizer': ['adam'],
        'learning_rate': [0.0005],
        'epochs': [100], #
        'batch_size': [64],
        'earlystop_patience': [15],
        'reduceLR_patience': [5],
        'num_workers': [6]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python main.py --use_crop --taskid %s "
              "--dataset_name %s --model %s --resize_height %s --resize_width %s --image_channels %s "
              "--first_out_channels %s --latent_dim %s "
              "--train_set_ratio %s --valid_set_ratio %s --optimizer %s --learning_rate %s "
              "--epochs %s --batch_size %s --earlystop_patience %s --reduceLR_patience %s "
              "--num_workers %s"
              % (taskid, temp['dataset_name'], temp['model'], temp['resize_height'], temp['resize_width'],
                 temp['image_channels'], temp['first_out_channels'], temp['latent_dim'],
                 temp['train_set_ratio'], temp['valid_set_ratio'], temp['optimizer'], temp['learning_rate'],
                 temp['epochs'], temp['batch_size'], temp['earlystop_patience'], temp['reduceLR_patience'],
                 temp['num_workers'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])