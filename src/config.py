# src/config.py

import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration for hyperparameters and settings.
    """
    train: bool
    # Paths


    #####################################################################################################
    experiment: int = 1
    model_type: str = 'GPD' # 'GEV' or 'GPD'
    #####################################################################################################

    root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    exp_dir: str = os.path.join(root_dir, 'experiments', f'{experiment}')
    data_dir: str = os.path.join(root_dir, 'data', f'{experiment}')
    models_dir: str = os.path.join(exp_dir, 'models')
    results_dir: str = os.path.join(exp_dir, 'results') 
    figures_dir: str = os.path.join(results_dir, 'figures')
    evaluation_dir: str = os.path.join(results_dir, 'evaluation')

    #####################################################################################################
    params_file_path: str = os.path.join(exp_dir,'params', f'{model_type}_params.txt')
    #####################################################################################################

    
    # Data files
    data_file: str = os.path.join(data_dir, 'precipitation_maxima.nc')
    ids_file: str = os.path.join(data_dir, 'ids.txt')
    #####################################################################################################


    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    n_train_percent: int         = 20 # NOTE Percentage of dataset 
    n_test_percent:  int         = 80

    noise_dim: int      = 100
    train_epoch: int    = 500
    decay_lab: int      = 90
    n_sub_ids: int      = 25
    smooth_factor: float = 0.1
    batch_size: int     = 50
    LAMBDA: float       = 0.1
    LAMB_epoch: int     = 0

    # lr and beta1 as specified in the DCGAN paper
    learning_rate: float = 0.0002 
    beta1: float        = 0.5
    
    D_train_it: int     = 2
    decay: float        = 0.9       # Exponential moving average decay
    moving_avg: bool = True
    batch_norm: bool = True

    # Options
    use_empirical_cdf: bool = True  # If False, use fitted GEV CDFs
    use_EC_regul: bool = False      # If True, use EC regularization

    @staticmethod
    def save_config(config, filename: str = "config.txt") -> None:
        """
        Save configuration parameters to a file.

        :param config: Configuration object.
        :param filename: Name of the file to save the configurations.
        """
        filepath = os.path.join(config.exp_dir, filename)
        with open(filepath, "w") as f:
            for attr, value in vars(config).items():
                f.write(f"{attr} = {value}\n")
