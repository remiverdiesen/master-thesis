# src/data_handler.py

import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.stats import rankdata, genextreme
from typing import Any, Tuple
from config import Config
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataHandler:
    """
    Handles data loading and preprocessing.
    """
    config: Config
    device: torch.device = field(init=False)
    data: Any = field(init=False, default=None)
    ids: np.ndarray = field(init=False, default=None)
    pos_coordinates: np.ndarray = field(init=False, default=None)
    params: np.ndarray = field(init=False, default=None)

    # Observations \mathbf{Z}_i
    Z_train: np.ndarray = field(init=False, default=None)   
    Z_test: np.ndarray = field(init=False, default=None)

    # Normalized margins \mathbf{U}_i
    U_train: np.ndarray = field(init=False, default=None)
    U_test: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.device = self.config.device
        self.prepare_data()

    def prepare_data(self) -> None:
        """
        Load and preprocess data.
        """
        try:
            # Load data
            self.data = xr.open_dataset(self.config.data_file)
            var_name = list(self.data.data_vars)[0]

            # Extract the desired variable values, assuming it's named 'Pr'
            Z_obs = self.data[var_name].values
            logger.debug(f"Loaded observation data with shape {Z_obs.shape}.") 

            # Load IDs
            self.ids = np.genfromtxt(self.config.ids_file, delimiter=',', skip_header=1, dtype=int)
            logger.debug(f"Loaded IDs with shape {self.ids.shape}.") 

            # Prepare data
            n_train = self.config.n_train_percent * Z_obs.shape[0] // 100
            n_test = Z_obs.shape[0] - n_train 

            # Split the dataset sequentially 
            train_set = Z_obs[:n_train, :, :]
            test_set  = Z_obs[n_train:, :, :]

            # Used in test.py for inverse transformation 
            self.Z_train = torch.tensor(train_set, dtype=torch.float32)
            self.Z_test  = torch.tensor(test_set, dtype=torch.float32)
            

            
            self.params = self.get_params(self.config, Z_obs.shape[1], Z_obs.shape[2])
            logger.debug(f"Got distribution to observations.")
      
            # Normalize margins
            if self.config.use_empirical_cdf:
                train_set = self.normalize_margins_empirical(train_set)
                test_set  = self.normalize_margins_empirical(test_set)
                # train_set min and max values
                logger.debug(f"Train set min: {train_set.min()}, max: {train_set.max()}")
                # test_set min and max values
                logger.debug(f"Test set min: {test_set.min()}, max: {test_set.max()}")
                logger.debug(f"Normalized margins using empirical CDF for train and test.")
            else:
                # NOTE: Alternatively, could use the fitted GEV distributions for normalization but this seems to give slightly worse results.
                train_set = self.normalize_margins_gev(train_set)
                test_set = self.normalize_margins_gev(test_set)
                logger.debug(f"Normalized margins using GEV CDF for train and test.")

            # Reshape data
            n_lat, n_lon = Z_obs.shape[1], Z_obs.shape[2]
            X_dim = [n_lat, n_lon, 1]

            train_set = train_set.reshape(-1, *X_dim)
            test_set = test_set.reshape(-1, *X_dim)
            logger.debug(f"Reshaped train to {train_set.shape}") 
            logger.debug(f"Reshaped test to  {test_set.shape}")  

            # Pad data
            pad = 1
            train_set = np.pad(train_set, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            test_set = np.pad(test_set, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            logger.debug(f"Padded train to   {train_set.shape}") 
            logger.debug(f"Padded test to    {test_set.shape}")  

            # Convert to torch tensors
            train_set = torch.tensor(train_set, dtype=torch.float32)
            test_set = torch.tensor(test_set, dtype=torch.float32)
            logger.debug("Converted data to torch tensors")

            # Rearrange dimensions from (N, H, W, C) to (N, C, H, W)
            train_set = train_set.permute(0, 3, 1, 2)
            test_set  = test_set.permute(0, 3, 1, 2)
            logger.debug(f"Train tensor shape: {train_set.shape}") 
            logger.debug(f"Test tensor shape:  {test_set.shape}")  

            # Move data to device
            self.U_train = train_set.to(self.device)
            self.U_test = test_set.to(self.device)
            logger.debug("Moved data to device.")

            # Adjust indices for padding
            pad_adjust = pad
            self.ids = self.ids + pad_adjust  # Adjust for padding
            self.ids = self.ids[:, [1, 0]]      # Now ids_ is in (latitude, longitude) order

            # Prepare position coordinates for EC computation
            self.pos_coordinates = self.pos_coords(self.config.n_sub_ids)
            logger.info("Data preparation complete!")

        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            raise e

    def get_params(self, config, n_lat: int, n_lon: int) -> np.ndarray:
        file_path = config.params_file_path
        # Initialize the GEV_params array
        params = np.zeros((n_lat, n_lon, 3))

        # Read the file and extract GEV parameters
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Parse each line in the format: (i, j): shape, loc, scale
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                grid_info, values = line.split(':')
                i, j = eval(grid_info)  # Get grid indices (1-based)
                shape, loc, scale = map(float, values.split(','))  # Get shape, loc, scale values
                
                # Store in GEV_params (convert 1-based to 0-based indexing)
                params[i-1, j-1, 0] = shape
                params[i-1, j-1, 1] = loc
                params[i-1, j-1, 2] = scale

        return params
    
    def normalize_margins_empirical(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the input data to uniform distribution using empirical CDF.
        """
        n_time, n_lat, n_lon = obs.shape
        uniform_data = np.zeros(obs.shape)

        for i in range(n_lat):
            for j in range(n_lon):
                time_series = obs[:, i, j]
                if np.all(time_series == time_series[0]):
                    uniform_data[:, i, j] = 0.5
                else:
                    ranks = rankdata(time_series, method='average')
                    uniform_data[:, i, j] = ranks / (n_time + 1)
        return uniform_data

    def normalize_margins_gev(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the input data to uniform margins using fitted GEV CDFs.
        """
        n_time, n_lat, n_lon = obs.shape
        uniform_data = np.zeros(obs.shape)

        for i in range(n_lat):
            for j in range(n_lon):
                shape, loc, scale = self.GEV_params[i, j, :]
                if np.isnan(shape):
                    uniform_data[:, i, j] = np.nan
                    continue
                time_series = obs[:, i, j]
                cdf_values = genextreme.cdf(time_series, c=shape, loc=loc, scale=scale)
                uniform_data[:, i, j] = cdf_values
        return uniform_data

    @staticmethod
    def pos_coords(n_sub_ids: int) -> np.ndarray:
        """
        Generate all unique pairs of indices for EC estimation.
        """
        indices = np.arange(n_sub_ids)
        vec = []
        for i in range(n_sub_ids):
            for j in range(i + 1, n_sub_ids):
                vec.append([i, j])
        return np.array(vec, dtype=int)

    def get_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the training and testing datasets.
        """
        return self.train_set, self.test_set
