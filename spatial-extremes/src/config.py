# src/config.py

import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Configuration parameters with default values
    train: bool = True
    experiment: str = '1'
    threshold: str = "4.4"
    season: str = "JJA" 
    period:str = "2010-2024"
    model_type: str = 'GEV'  # 'GEV' 'eGPD'or 'GPD'
    input_data_file = f'precip_10km_2010-2024_{season}_24h.nc'
    
    if experiment == '1':
        # Directories (initialized with defaults, will be overridden if reading from config file)
        root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Paths relative to the root directory
        exp_dir: str = os.path.join(root_dir, 'experiments')
        data_dir: str = os.path.join(root_dir, 'data')
        
        # Paths specific to the experiment number
        experiment_dir: str = os.path.join(exp_dir, str(experiment))
        experiment_data_dir: str = os.path.join(data_dir, experiment)
        
        # Model and results directories specific to the experiment
        models_dir: str = os.path.join(experiment_dir,  'model', model_type)
        results_dir: str = os.path.join(experiment_dir,  'results')
        figures_dir: str = os.path.join(experiment_dir, 'figures')
        evaluation_dir: str = os.path.join(experiment_dir, 'evaluation')

        # Parameter file paths
        params_file_path: str = os.path.join(experiment_dir, 'params', f'{model_type}_params.txt')

        # Data files
        data_file: str = os.path.join(experiment_data_dir, 'precipitation_maxima.nc')
        ids_file: str = os.path.join(data_dir, experiment, 'ids.txt')

    else:
        # Directories (initialized with defaults, will be overridden if reading from config file)
        root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Paths relative to the root directory
        exp_dir: str = os.path.join(root_dir, 'experiments')
        data_dir: str = os.path.join(root_dir, 'data')
        
        # Paths specific to the experiment number
        experiment_dir: str = os.path.join(exp_dir, str(experiment))
        experiment_data_dir: str = os.path.join(data_dir, data_dir, experiment, period, season)
        
        # Model and results directories specific to the experiment
        models_dir: str = os.path.join(experiment_dir, period, season,  'model')
        results_dir: str = os.path.join(experiment_dir, period, season, 'results')
        figures_dir: str = os.path.join(experiment_dir, period, season, 'figures')
        evaluation_dir: str = os.path.join(experiment_dir, period, season,'evaluation')

        # Parameter file paths
        params_file_path: str = os.path.join(experiment_dir, period, season,  f'{model_type}_params_interpolated_threshold_{threshold}.txt')

        # Data files
        data_file: str = os.path.join(experiment_data_dir, input_data_file)
        ids_file: str = os.path.join(data_dir, experiment, 'ids.txt')

    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    n_train_percent: int = 80  # Percentage of dataset
    n_test_percent: int = 20
    noise_dim: int = 100
    train_epoch: int = 500
    decay_lab: int = 90
    n_sub_ids: int = 25
    smooth_factor: float = 0.1
    batch_size: int = 50
    LAMBDA: float = 0.1
    LAMB_epoch: int = 0

    # Learning rate and parameters as per DCGAN paper
    learning_rate: float = 0.002 # 0 0.0002
    beta1: float = 0.5

    # Model parameters
    D_train_it: int = 2
    decay: float = 0.9  # Exponential moving average decay
    moving_avg: bool = True
    batch_norm: bool = True

    # Options
    use_empirical_cdf: bool = False  # Use empirical CDF if True
    use_EC_regul: bool = False      # Use EC regularization if True

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

    @staticmethod
    def read_config(filepath: str) -> 'Config':
        """
        Read configuration parameters from a file to a Config object.

        :param filepath: Path to the configuration file.
        :return: Config object with parameters loaded from file.
        """
        config_dict = {}
        with open(filepath, "r") as f:
            for line in f:
                # Assuming the format 'attr = value'
                key, value = line.strip().split(' = ', 1)
                
                # Convert string values to correct types
                if value.isdigit():
                    config_dict[key] = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    config_dict[key] = float(value)
                elif value.lower() == 'true':
                    config_dict[key] = True
                elif value.lower() == 'false':
                    config_dict[key] = False
                else:
                    config_dict[key] = value
        
        return Config(**config_dict)

        # Load configuration
        # Load configuration directly from a config file path
    @classmethod
    def load_config(cls, train: bool, file_path: str) -> 'Config':
        """
        Load configuration from a given config file.

        :param train: Boolean indicating if in training mode.
        :param file_path: Path to the configuration file.
        :return: Config object.
        """
        config = cls(train=train)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        if hasattr(config, key):
                            # Set the attribute based on the key from the file
                            setattr(config, key, value.strip())
        else:
            raise FileNotFoundError(f"Configuration file not found at {file_path}")
        
        return config
