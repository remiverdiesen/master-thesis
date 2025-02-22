import numpy as np
import netCDF4 as nc
import torch

def load_netcdf(file_path):
    """Load precipitation maxima from NetCDF file."""
    ds = nc.Dataset(file_path)
    precip_max = ds.variables['precip_max'][:]  # Shape: [time, lat, lon]
    ds.close()
    return precip_max  # [n_samples, height, width]

def ecdf_transform(data):
    """Transform data to uniform margins using empirical CDF."""
    n, h, w = data.shape
    data_flat = data.reshape(n, -1)  # [n, h*w]
    u_flat = np.zeros_like(data_flat, dtype=np.float32)
    for j in range(data_flat.shape[1]):
        if np.all(data_flat[:, j] == data_flat[0, j]):  # Handle constant data
            u_flat[:, j] = 0
        else:
            ranks = np.argsort(np.argsort(data_flat[:, j])) + 1
            u_flat[:, j] = ranks / (n + 1)  # Uniform in (0,1)
    return u_flat.reshape(n, h, w, 1)

def pad_data(data, pad_width=1):
    """Pad data with zeros for GAN input size."""
    return np.pad(data, ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0)), 
                  mode='constant', constant_values=0)

def load_and_preprocess_data(config):
    """Load, transform, and pad training data."""
    precip_max = load_netcdf(config['data']['train_file'])  # [n_samples, 18, 22]
    u_train = ecdf_transform(precip_max)  # [n_samples, 18, 22, 1]
    u_train_padded = pad_data(u_train)  # [n_samples, 20, 24, 1]
    return torch.tensor(u_train_padded, dtype=torch.float32), precip_max