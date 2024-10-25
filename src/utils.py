# src/utils.py

import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from numpy.linalg import norm
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import sqrtm

import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging
import matplotlib.pyplot as plt  # Add this import for plotting

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
def inverse_gpd(uniform, params):
    """
    Apply the inverse GPD transformation for given uniform values and GPD parameters.
    
    Parameters:
    - uniform: array of uniform values in the open interval (0, 1)
    - params: GPD parameters [xi, sigma, threshold]
    
    Returns:
    - inv: Inverse transformed values on the original scale.
    """

    sigma, threshold, xi  = params

    # Ensure uniform values are in the open interval (0, 1)
    epsilon = np.finfo(float).eps  # Smallest positive float number
    uniform = np.clip(uniform, epsilon, 1 - epsilon)

    # Handle GPD case where xi == 0
    if xi == 0:
        inv = threshold + sigma * (-np.log(1 - uniform))
    else:
        inv = threshold + (sigma / xi) * ((1 - uniform) ** (-xi) - 1)
    return inv

def inverse_gpd_(dat_, params_):
    """
    Apply inverse GPD transformation to an entire dataset.
    
    Parameters:
    - dat_: Generated uniform samples (n_samples, n_points)
    - params_: GPD parameters for each point (n_points, 3)
    
    Returns:
    - inversed_gpd: Inverse transformed values in original scale.
    - min_unif, max_unif: Minimum and maximum uniform values.
    - min_inv, max_inv: Minimum and maximum inverse-transformed values.
    """
    # Apply empirical CDF (uniform transformation)
    uniform = np.apply_along_axis(
        lambda x: (np.argsort(np.argsort(x)) + 1) / (len(x) + 1), axis=0, arr=dat_
    )

    min_unif = np.min(uniform)
    max_unif = np.max(uniform)

    # Debug the uniform output
    assert 0 <= min_unif <= 1 and 0 <= max_unif <= 1, f"Uniform values out of range: [{min_unif}, {max_unif}]"

    # Initialize inversed_gpd array
    inversed_gpd = np.zeros_like(uniform)

    # Apply inverse GPD transformation column-wise
    for i in range(uniform.shape[1]):
        inversed_gpd[:, i] = inverse_gpd(uniform[:, i], params_[i, :])

    min_inv = np.min(inversed_gpd)
    max_inv = np.max(inversed_gpd)

    # Check for negative values (GPD can be negative depending on xi)
    if np.any(inversed_gpd < 0):
        print(f"Warning: Negative inverse-transformed values detected. Min value: {min_inv}")

    return inversed_gpd, min_unif, max_unif, min_inv, max_inv

def inverse_gev(uniform, params):
    """
    Apply the inverse GEV transformation for given uniform values and GEV parameters.
    
    Parameters:
    - uniform: array of uniform values (0, 1)
    - params: GEV parameters [xi, sigma, mu]
    
    Returns:
    - inverse_transformed: Inverse transformed values on the original scale.
    """
    xi, sigma, mu = params
    
    # Handle GEV case where xi == 0
    if xi == 0:
        inv = mu - sigma * np.log(-np.log(uniform))
    else:
        inv = (((-np.log(uniform))**(-xi) - 1) * sigma / xi) + mu
    return inv

def inverse_gev_(dat_, params_):
    """
    Apply inverse GEV transformation to an entire dataset.
    
    Parameters:
    - dat_: Generated uniform samples (n_samples, n_points)
    - params_: GEV parameters for each point (n_points, 3)
    
    Returns:
    - inversed_gev: Inverse transformed values in original scale.
    - min_unif, max_unif: Minimum and maximum uniform values.
    - min_inv, max_inv: Minimum and maximum inverse-transformed values.
    """
    # Apply empirical CDF (uniform transformation)
    uniform = np.apply_along_axis(lambda x: (np.argsort(np.argsort(x)) + 1) / (len(x) + 1), axis=0, arr=dat_)
    
    min_unif = np.min(uniform)
    max_unif = np.max(uniform)
    
    # Debug the uniform output
    assert min_unif >= 0 and max_unif <= 1, f"Uniform values out of range: [{min_unif}, {max_unif}]"

    # Initialize inversed_gev array
    inversed_gev = np.zeros_like(uniform)
    
    # Apply inverse GEV transformation column-wise
    for i in range(uniform.shape[1]):
        inversed_gev[:, i] = inverse_gev(uniform[:, i], params_[i, :])
    
    min_inv = np.min(inversed_gev)
    max_inv = np.max(inversed_gev)

    assert min_inv < 0, f"Unrealisitc minimum value, inverse-transformed value is negative: {min_inv}"

    return inversed_gev, min_unif, max_unif, min_inv, max_inv

def inverse_ecdf(dat_, train):
    """
    Apply inverse ECDF transformation to the generated data using the empirical distribution of the training data.
    
    Parameters:
    - dat_:  Generated uniform samples (n_samples, n_points)
    - train: Training data (n_points, n_observations) from which quantiles are computed.
    
    Returns:
    - inversed_ecdf: Inverse transformed values based on the empirical CDF.
    """
    # Apply empirical CDF (uniform transformation)
    uniform = (np.argsort(np.argsort(dat_)) + 1) / (len(dat_) + 1)

    # Map uniform values to corresponding quantiles in the training data
    inversed_ecdf = np.quantile(train, uniform)
    
    return inversed_ecdf

def inverse_transform(U_samples: np.ndarray, Z_train: np.ndarray, params: np.ndarray, ids_: np.ndarray, config) -> np.ndarray:
    """
    Transform generated samples back to original scale using inverse GEV CDFs OR empirical CDF.
    
    Parameters:
    - U_samples: Generated uniform samples (n_samples, n_lat, n_lon)
    - Z_train: Training data (n_train, n_lat, n_lon) for inverse ECDF transformation
    - params:  parameters (n_lat, n_lon, 3) for inverse GEV transformation
    - ids_: Grid point indices (not used directly in this version)
    - use_empirical_cdf: Boolean flag to choose between empirical CDF and GEV transformation
    
    Returns:
    - Z_generated: Generated samples in the original scale (n_samples, n_lat, n_lon)
    """
    num_samples = U_samples.shape[0]
    n_lat, n_lon = params.shape[0], params.shape[1]

    logger.debug(f"Transforming {num_samples} samples back to original scale...")

    # To store observations in original scale
    Z_generated = np.zeros_like(U_samples)

    # Inverse transform the generated samples back to the original scale
    if config.use_empirical_cdf:
        # Apply inverse ECDF transformation (DCGAN)
        for i in range(n_lat):
            for j in range(n_lon):
                Z_generated[:, i, j] = inverse_ecdf(U_samples[:, i, j], Z_train[:, i, j])
    elif config.model_type = 'GEV'
        # Apply inverse GEV transformation (evtGAN)
        for i in range(n_lat):
            for j in range(n_lon):
                Z_generated[:, i, j], _, _, _, _ = inverse_gev_(U_samples[:, i, j], arams[i, j, :])
    
    else config.model_type = 'GPD'
        # Apply inverse GEV transformation (evtGAN)
        for i in range(n_lat):
            for j in range(n_lon):
                Z_generated[:, i, j], _, _, _, _ = inverse_gpd_(U_samples[:, i, j], params[i, j, :])
    return Z_generated

def weights_init(m: nn.Module) -> None:
    """
    Initialize weights of the network.
    """
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    """
    Update the Exponential Moving Average (EMA) model parameters.
    """
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name in model_params.keys():
            ema_params[name].data.mul_(decay).add_(model_params[name].data * (1 - decay))

def compute_EC_loss(fake_data: torch.Tensor, real_data: torch.Tensor, ids_: np.ndarray,
                    pos_coordinates: np.ndarray, config) -> torch.Tensor:
    """
    Compute the EC loss between fake data and real data.
    """
    try:
        # Sample subset of indices for EC computation
        n_sub_ids = config.n_sub_ids
        all_indices = np.arange(len(ids_))
        selected_indices = np.random.choice(all_indices, n_sub_ids, replace=False)
        ind_loss = ids_[selected_indices]

        # Ensure indices are integers
        ind_loss = ind_loss.astype(int)

        # Get the data at these indices
        fake_sub = fake_data[:, 0, ind_loss[:, 0], ind_loss[:, 1]]  # Shape: (batch_size, n_sub_ids)
        real_sub = real_data[:, 0, ind_loss[:, 0], ind_loss[:, 1]]

        # Ensure values are in (0,1)
        fake_sub = torch.clamp(fake_sub, min=1e-8, max=1 - 1e-8)
        real_sub = torch.clamp(real_sub, min=1e-8, max=1 - 1e-8)

        # Transform to exponential margins
        fake_exp = -torch.log(1 - fake_sub)  # Shape: (batch_size, n_sub_ids)
        real_exp = -torch.log(1 - real_sub)

        # Compute ECs
        EC_fake = compute_ECs_from_exp(fake_exp, pos_coordinates).to(config.device)
        EC_real = compute_ECs_from_exp(real_exp, pos_coordinates).to(config.device)

        # Compute EC loss (e.g., Mean Squared Error)
        # EC_loss = torch.norm(EC_fake - EC_real) / torch.sqrt(torch.tensor(len(EC_real), dtype=torch.float32, device=config.device))
        EC_loss = torch.mean((EC_fake - EC_real) ** 2)

        return EC_loss
    except Exception as e:
        logger.error(f"Error in compute_EC_loss: {e}")
        raise e

def compute_ECs_from_exp(exp_data: torch.Tensor, pos_coordinates: np.ndarray) -> torch.Tensor:
    """
    Compute ECs from exponential data.
    """
    n_samples = exp_data.shape[0]
    idx1 = pos_coordinates[:, 0]
    idx2 = pos_coordinates[:, 1]

    E1 = exp_data[:, idx1]  # (batch_size, n_pairs)
    E2 = exp_data[:, idx2]  # (batch_size, n_pairs)

    # Calculate Minima Across Pairs of Locations
    minima = torch.min(E1, E2)  # (batch_size, n_pairs)

    # Estimate ECs Using Empirical Method
    sum_minima = torch.sum(minima, dim=0)  # Sum over batch dimension, shape: (n_pairs,)
    epsilon = 1e-6
    sum_minima = sum_minima + epsilon

    ECs = n_samples / sum_minima  # Shape: (n_pairs,)

    return ECs

def inv_unit_frechet(uniform: torch.Tensor) -> torch.Tensor:
    """
    Transform uniform distribution to unit Fréchet distribution using PyTorch.

    :param uniform: PyTorch tensor with uniform values in (0,1).
    :return: PyTorch tensor with unit Fréchet distributed values.
    """
    logger.debug(f"Applying inverse unit Fréchet transformation on data of shape {uniform.shape}.")
    eps = 1e-10
    uniform = torch.clamp(uniform, min=eps, max=1 - eps)
    result = -1 / torch.log(uniform)
    logger.debug("Inverse unit Fréchet transformation complete.")
    return result

def plot_loss_curve(generator_losses, discriminator_losses, config):
    """Plots and saves the generator and discriminator loss curves."""
    plt.figure()
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Save the figure
    figures_dir = config.figures_dir
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, f'loss_curve_epoch{config.train_epoch}.png'))
 
def ks_test(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Perform KS test to compare marginal distributions of real and generated samples.
    """
    n_lat, n_lon = Z_generated.shape[1], Z_generated.shape[2]
    ks_list = []
    ks_statistics = []

    # Loop over grid points (lat, lon) and perform KS test for each dimension
    for i in range(n_lat):
        for j in range(n_lon):
            ks_stat, p_value = ks_2samp(Z_test[:, i, j], Z_generated[:, i, j])
            ks_list.append(f'{i},{j}  -  ks_stat: {ks_stat}, p_value: {p_value}')
            ks_statistics.append(ks_stat)

    # Return the average KS statistic across grid points
    return np.mean(ks_statistics), ks_list

def chi_statistic( Z_generated: np.ndarray, Z_test: np.ndarray, threshold: float = 0.95) -> float:
    """
    Compute Chi statistic to measure extremal dependence in real and generated data.
    """
    n_lat, n_lon = Z_test.shape[1], Z_test.shape[2]
    chi_stats = []
    
    for i in range(n_lat):
        for j in range(n_lon):
            real_extremes = Z_test[:, i, j] > np.quantile(Z_test[:, i, j], threshold)
            generated_extremes = Z_generated[:, i, j] > np.quantile(Z_generated[:, i, j], threshold)
            
            # Joint exceedances in real and generated data
            joint_exceedances = np.sum(real_extremes & generated_extremes)
            total_exceedances = np.sum(real_extremes) + np.sum(generated_extremes)
            
            chi_stat = (joint_exceedances / total_exceedances)
            chi_stats.append(chi_stat)
    
    return np.mean(chi_stats)

def mean_squared_error(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between generated and real data.
    
    Parameters:
    - Z_generated: Generated samples (n_samples, n_lat, n_lon)
    - Z_test: Test (real) samples (n_samples, n_lat, n_lon)
    
    Returns:
    - mse: Mean Squared Error value.
    """
    # Ensure the datasets have the same number of samples
    n_samples = min(Z_generated.shape[0], Z_test.shape[0])
    
    # Compute the MSE across all samples, latitudes, and longitudes
    mse = np.mean((Z_generated[:n_samples] - Z_test[:n_samples]) ** 2)
    return mse

def frechet_inception_distance(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID) between real and generated data.
    
    Parameters:
    - Z_generated: Generated samples (n_samples, n_lat, n_lon)
    - Z_test: Test (real) samples (n_samples, n_lat, n_lon)
    
    Returns:
    - fid: Fréchet Inception Distance value.
    """
    # Flatten the spatial dimensions for each sample
    Z_generated_flat = Z_generated.reshape(Z_generated.shape[0], -1)
    Z_test_flat = Z_test.reshape(Z_test.shape[0], -1)
    
    # Compute the mean and covariance of real and generated data
    mu_real = np.mean(Z_test_flat, axis=0)
    mu_generated = np.mean(Z_generated_flat, axis=0)
    
    cov_real = np.cov(Z_test_flat, rowvar=False)
    cov_generated = np.cov(Z_generated_flat, rowvar=False)
    
    # Compute the mean difference
    mu_diff = norm(mu_real - mu_generated)
    
    # Add a small epsilon to the diagonal of the covariance matrices to prevent numerical instability
    epsilon = 1e-6
    cov_real += np.eye(cov_real.shape[0]) * epsilon
    cov_generated += np.eye(cov_generated.shape[0]) * epsilon
    
    # Compute the square root of the product of cov_real and cov_generated using scipy's sqrtm
    cov_sqrt, _ = sqrtm(np.dot(cov_real, cov_generated), disp=False)
    
    # If there are complex values due to numerical errors, take only the real part
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    # Compute the Fréchet distance
    fid = mu_diff**2 + np.trace(cov_real + cov_generated - 2 * cov_sqrt)
    
    return fid

def inception_score(Z_generated: np.ndarray, netG, config) -> float:
    """
    Compute Inception Score for generated samples.
    
    Parameters:
    - Z_generated: Generated samples (n_samples, n_lat, n_lon)
    - netG: Pre-trained model to extract features for scoring
    - device: Torch device (e.g., 'cpu' or 'cuda')
    - batch_size: Batch size for processing
    
    Returns:
    - inception_score: Inception Score for the generated data.
    """
    preds = []
    Z_generated_tensor = torch.tensor(Z_generated).to(config.device)
    num_samples = Z_generated_tensor.shape[0]
    
    # Loop over batches of generated samples
    for i in range(0, num_samples, config.batch_size):
        batch = Z_generated_tensor[i:i + config.batch_size]
        
        # Generate noise for the generator and create fake data
        noise = torch.randn(batch.shape[0], config.noise_dim, device=config.device)
        pred = netG(noise)  # Generate samples using the generator
        
        # Apply softmax to convert logits to class probabilities and store them
        preds.append(F.softmax(pred, dim=-1).detach().cpu().numpy())

    
    # Combine all predictions into one array
    preds = np.concatenate(preds, axis=0)
    
    # Compute the mean prediction over all samples
    mean_preds = np.mean(preds, axis=0)
    
    # Calculate the KL divergence
    kl_div = preds * (np.log(preds) - np.log(mean_preds))
    
    # Compute the Inception Score
    inception_score = np.exp(np.mean(np.sum(kl_div, axis=1)))
    
    return inception_score

def wasserstein_distance_grid(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Compute Wasserstein Distance between real and generated samples.
    
    Parameters:
    - Z_generated: Generated samples (n_samples, n_lat, n_lon)
    - Z_test: Test (real) samples (n_samples, n_lat, n_lon)
    
    Returns:
    - avg_wd: Average Wasserstein distance across all grid points.
    """
    n_lat, n_lon = Z_generated.shape[1], Z_generated.shape[2]
    wd_stats = []
    
    # Loop over grid points (lat, lon) and compute Wasserstein distance for each dimension
    for i in range(n_lat):
        for j in range(n_lon):
            wd = wasserstein_distance(Z_test[:, i, j], Z_generated[:, i, j])
            wd_stats.append(wd)
    
    # Return the average Wasserstein distance across grid points
    return np.mean(wd_stats)

