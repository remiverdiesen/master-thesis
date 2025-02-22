import numpy as np
import xarray as xr
import torch

def load_netcdf(file_path):
    """Load precipitation maxima from NetCDF file."""
    ds = xr.open_dataset(file_path)
    var_name = list(ds.data_vars)[0]  # Assumes first variable is the target
    precip_max = ds[var_name].values  # Convert to NumPy array [n_samples, height, width]
    ds.close()
    return precip_max

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
    return u_flat.reshape(n, h, w)

def pad_data(data, pad_width=1):
    """Pad data with zeros for GAN input size."""
    return np.pad(data, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 
                  mode='constant', constant_values=0)

def load_and_preprocess_data(config):
    """Load, transform, and pad training data."""
    precip_max = load_netcdf(config['data']['train_file'])  # [n_samples, 18, 22]
    u_train = ecdf_transform(precip_max)                    # [n_samples, 18, 22]
    u_train_padded = pad_data(u_train)                      # [n_samples, 20, 24]
    # Convert to tensor and add channel dimension
    u_train_tensor = torch.tensor(u_train_padded, dtype=torch.float32).unsqueeze(1)  # [n_samples, 1, 20, 24]
    return u_train_tensor, precip_max