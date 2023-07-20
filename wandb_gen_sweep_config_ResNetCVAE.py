import math
import yaml

def main():
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'Validation Loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {

        'resize': {
            # change values here: [128, 256, 512] for VanillaCVAE, [64, 128, 256, 512] for ResNetCVAE
            # 'distribution': 'constant',
            'values': [128, 256, 512]
        },

        'latent_dim': {
            #'distribution': 'categorical',
            'values': [128, 256, 512, 1024]
        },

        'first_out_channels': {
            # change values here: 4, 8, 16 for VanillaCVAE; 64,128 for ResNetCVAE
            # 'distribution': 'constant',
            'values': [8, 16, 32]
        }
    }

    parameters_dict.update({
        'seed': {
            #'distribution': 'constant',
            'value': 42
        },

        'dataset_name': {
            #'distribution': 'categorical',
            'values': ['BRAIX']
        },

        'save_dir': {
            #'distribution': 'categorical',
            'values': ['./output/']
        },

        'model': {
            # change values here: VanillaCVAE or ResNetCVAE
            #'distribution': 'categorical',
            'values': ['ResNetCVAE']
        },

        'imageinfo_clinical_path':{
            #'distribution': 'categorical',
            'values': ['./data/Hui_BRAIX_dataset_info.xlsx']
        },

        'learning_rate': {
            # a flat distribution between 0 and 0.01
            #'distribution': 'constant',
            'value': 0.0005
        },

        'epochs': {
            #'distribution': 'constant',
            'value': 100
        },

        'batch_size': {
            #'distribution': 'constant',
            'value': 64
        },

        'train_set_ratio': {
            #'distribution': 'constant',
            'value': 0.6
        },

        'valid_set_ratio': {
            #'distribution': 'constant',
            'value': 0.1
        },

        'num_workers': {
            #'distribution': 'constant',
            'value': 6
        },

        'image_channels': {
            #'distribution': 'constant',
            'value': 1
        },

        'num_Blocks':{
            # do not need to be deleted here, will not be used for VanillaCVAE
            #'distribution': 'categorical',
            'values':['2,2,2,2']
        },

        'optimizer': {
            #'distribution': 'categorical',
            'values': ['adam']
        },

        'earlystop_patience': {
            #'distribution': 'constant',
            'value': 15
        },

        'reduceLR_patience': {
            #'distribution': 'constant',
            'value': 5
        }
    })
    sweep_config['parameters'] = parameters_dict

    with open('wandb_sweep_config_ResNetCVAE.yaml', 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)

if __name__ == '__main__':
    main()