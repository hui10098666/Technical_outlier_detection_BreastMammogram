import math
import yaml


def main():
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'Validation Loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'resize_height': {
            # change values here: [128, 256, 512] for VanillaCVAE, [64, 128, 256, 512] for ResNetCVAE
            'distribution': 'categorical',
            'values': [128, 256, 512]
        },

        'resize_width': {
            # change values here: [128, 256, 512] for VanillaCVAE, [64, 128, 256, 512] for ResNetCVAE
            'distribution': 'categorical',
            'values': [128, 256, 512]
        },

        'latent_dim': {
            'distribution': 'categorical',
            'values': [128, 256, 384]
        },

        'learning_rate': {
            # a flat distribution between 0 and 0.01
            'distribution': 'uniform',
            'min': 0,
            'max': 0.01
        },

        'epochs': {
            'distribution': 'categorical',
            'values': [100, 150, 200]
        },

        'batch_size': {
            'distribution': 'categorical',
            'values': [64, 128, 256]
        },

        'first_out_channels': {
            # change values here: 4, 8, 16 for VanillaCVAE; 64,128 for ResNetCVAE
            'distribution': 'categorical',
            'values': [4, 8, 16]
        }
    }

    parameters_dict.update({
        'seed': {
            'distribution': 'constant',
            'value': 42
        },

        'dataset_name': {
            'distribution': 'categorical',
            'values': ['BRAIX']
        },

        'save_dir': {
            'distribution': 'categorical',
            'values': ['./output/']
        },

        'model': {
            # change values here: VanillaCVAE or ResNetCVAE
            'distribution': 'categorical',
            'values': ['VanillaCVAE']
        },

        'tune_hyperparameter': {
            'distribution': 'categorical',
            'values': [True]
        },

        'imageinfo_clinical_path': {
            'distribution': 'categorical',
            'values': ['./data/Hui_BRAIX_subset_info.xlsx']
        },

        'train_set_ratio': {
            'distribution': 'constant',
            'value': 0.8
        },

        'valid_set_ratio': {
            'distribution': 'constant',
            'value': 0.1
        },

        'num_workers': {
            'distribution': 'constant',
            'value': 6
        },

        'image_channels': {
            'distribution': 'constant',
            'value': 1
        },

        'optimizer': {
            'distribution': 'categorical',
            'values': ['adam']
        },

        'earlystop_patience': {
            'distribution': 'constant',
            'value': 15
        },

        'reduceLR_patience': {
            'distribution': 'constant',
            'value': 5
        }
    })
    sweep_config['parameters'] = parameters_dict

    with open('wandb_sweep_config_VanillaCVAE.yaml', 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

if __name__ == '__main__':
    main()