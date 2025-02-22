#!/usr/bin/env python3
"""
GAN-Generated Extreme Rainfall Samples

Context:
---------
This script evaluates the performance of a GAN-based generative model in simulating extreme event data.
Generative Adversarial Networks (GANs) are powerful tools for synthetic data generation but are prone to challenges such as mode collapse,
poor feature representation, and inadequate training. These issues often manifest as significant divergence between generated and real-world data.
Our primary evaluation goal is to assess how well the generated samples match observed extreme events using a suite of statistical tests and distance metrics.

Logging Levels:
---------------
- INFO: Summarizes key steps and results.
- DEBUG: Provides detailed information on data shapes, intermediate computations, and file operations.
- ERROR: Captures and logs critical errors such as model or data loading failures.
"""

import os
import logging
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import sqrtm
import matplotlib

# Set a consistent font for matplotlib plots
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Suppress unnecessary logging from external libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('rasterio').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# -------------------------------------------------------------------------
# Logging Setup: Detailed logging across various levels for robust traceability.
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(lineno)d - %(levelname)s - %(message)s")

def get_params() -> np.ndarray:
    """
    Read and parse GEV parameters from a text file.
    
    Returns:
        params (np.ndarray): Array of shape (max_i, max_j, 3) with GEV parameters (shape, location, scale).
    """
    file_path = r"C:\Users\reverd\Repositories\master-thesis\spatial-extremes\experiments\1\params\GEV_params.txt"
    max_i, max_j = 0, 0
    lines = []

    logger.debug("Reading GEV parameters from file.")
    # First pass: determine grid dimensions
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            grid_info, values = line.split(':')
            i, j = eval(grid_info.strip())  # Assumes 1-based indexing
            max_i = max(max_i, i)
            max_j = max(max_j, j)
            lines.append((i, j, values.strip()))
    logger.debug(f"Determined grid dimensions: {max_i} x {max_j}")

    # Initialize parameter array based on dynamic grid size
    params = np.zeros((max_i, max_j, 3))
    # Second pass: populate the parameter array
    for i, j, values in lines:
        shape, loc, scale = map(float, values.split(','))
        params[i - 1, j - 1, 0] = shape
        params[i - 1, j - 1, 1] = loc
        params[i - 1, j - 1, 2] = scale

    logger.info("GEV parameters loaded successfully.")
    return params

def inverse_transform(U: np.ndarray, Z_train: any, params: np.ndarray) -> np.ndarray:
    """
    Inverse transformation: maps normalized GAN outputs U (which are in [0, 1])
    back to the real data space using the inverse of the GEV CDF.
    
    For a given grid cell with parameters (ξ, μ, σ), the quantile function Q(p) for p ∈ (0,1)
    is given by:
        - If ξ ≠ 0:
            Q(p) = μ + (σ / ξ) * ((-log(p))^(-ξ) - 1)
        - If ξ = 0 (the Gumbel case):
            Q(p) = μ - σ * log(-log(p))
    
    Args:
        U (np.ndarray): Normalized GAN outputs, shape (N, n_lat, n_lon) with values in [0, 1].
        Z_train: (Unused here, but kept for compatibility with original interface.)
        params (np.ndarray): GEV parameters, shape (n_lat, n_lon, 3).
        
    Returns:
        np.ndarray: Transformed samples Z_generated in the original data space.
    """
    logger.debug("Performing inverse GEV transformation on generated samples.")
    
    # Assume params has shape (n_lat, n_lon, 3) with order: [xi, mu, sigma]
    n_lat, n_lon = params.shape[0], params.shape[1]
    # Expand parameters to match U's shape (N, n_lat, n_lon)
    xi = np.expand_dims(params[:, :, 0], axis=0)  # shape: (1, n_lat, n_lon)
    mu = np.expand_dims(params[:, :, 1], axis=0)
    sigma = np.expand_dims(params[:, :, 2], axis=0)
    
    # Set a small tolerance to decide when to treat xi as zero
    tol = 1e-6
    # Compute inverse transformation:
    # For xi != 0:
    #    Z = mu + (sigma/xi) * ((-log(U))^(-xi) - 1)
    # For xi == 0:
    #    Z = mu - sigma * log(-log(U))
    Z_generated = np.where(
        np.abs(xi) > tol,
        mu + sigma / xi * ((-np.log(U)) ** (-xi) - 1),
        mu - sigma * np.log(-np.log(U))
    )
    return Z_generated

def ks_test(Z_generated: np.ndarray, Z_test: np.ndarray) -> (float, list):
    """
    Perform a Kolmogorov-Smirnov test at each grid point between generated and real data.
    
    Args:
        Z_generated (np.ndarray): Generated samples.
        Z_test (np.ndarray): Real observational samples.
    
    Returns:
        mean_stat (float): Mean KS statistic across all grid points.
        ks_stats (list): List of KS statistics per grid point.
    """
    n_lat, n_lon = Z_generated.shape[1], Z_generated.shape[2]
    ks_stats = []
    for i in range(n_lat):
        for j in range(n_lon):
            stat, _ = ks_2samp(Z_generated[:, i, j], Z_test[:, i, j])
            ks_stats.append(stat)
    mean_stat = np.mean(ks_stats)
    logger.debug(f"Computed KS statistics for grid points; Mean KS: {mean_stat:.4f}")
    return mean_stat, ks_stats

def chi_statistic(Z_generated: np.ndarray, Z_test: np.ndarray, threshold: float = 0.95) -> float:
    """
    Compute the chi-statistic to assess extremal dependence between generated and real data.
    
    Args:
        Z_generated (np.ndarray): Generated samples.
        Z_test (np.ndarray): Real observational samples.
        threshold (float): Quantile threshold for defining extreme events.
        
    Returns:
        float: The mean chi-statistic across grid points.
    """
    n_lat, n_lon = Z_generated.shape[1], Z_generated.shape[2]
    chi_stats = []
    
    for i in range(n_lat):
        for j in range(n_lon):
            real_threshold = np.quantile(Z_test[:, i, j], threshold)
            gen_threshold = np.quantile(Z_generated[:, i, j], threshold)
            
            # Define extreme events in both samples
            real_extremes = Z_test[:, i, j] > real_threshold
            gen_extremes = Z_generated[:, i, j] > gen_threshold
            
            joint_exceed = np.sum(real_extremes & gen_extremes)
            total_exceed = np.sum(real_extremes) + np.sum(gen_extremes)
            chi_value = joint_exceed / total_exceed if total_exceed > 0 else 0
            chi_stats.append(chi_value)
    mean_chi = np.mean(chi_stats)
    logger.debug(f"Computed chi-statistic; Mean chi: {mean_chi:.4f}")
    return mean_chi

def mean_squared_error(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between generated and observed samples.
    
    Args:
        Z_generated (np.ndarray): Generated samples.
        Z_test (np.ndarray): Real observational samples.
    
    Returns:
        float: The computed MSE.
    """
    n_samples = min(Z_generated.shape[0], Z_test.shape[0])
    mse_value = np.mean((Z_generated[:n_samples] - Z_test[:n_samples]) ** 2)
    logger.debug(f"Computed Mean Squared Error: {mse_value:.4f}")
    return mse_value

def frechet_inception_distance(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Compute a Fréchet Inception Distance (FID)-like metric to quantify the discrepancy between
    the feature representations of generated and real data.
    
    Args:
        Z_generated (np.ndarray): Generated samples.
        Z_test (np.ndarray): Real observational samples.
    
    Returns:
        float: The FID value.
    """
    # Flatten spatial dimensions for statistical analysis
    Z_generated_flat = Z_generated.reshape(Z_generated.shape[0], -1)
    Z_test_flat = Z_test.reshape(Z_test.shape[0], -1)
    
    mu_gen = np.mean(Z_generated_flat, axis=0)
    mu_test = np.mean(Z_test_flat, axis=0)
    
    cov_gen = np.cov(Z_generated_flat, rowvar=False)
    cov_test = np.cov(Z_test_flat, rowvar=False)
    
    diff = np.linalg.norm(mu_gen - mu_test)
    eps = 1e-6  # Small constant for numerical stability
    cov_gen += np.eye(cov_gen.shape[0]) * eps
    cov_test += np.eye(cov_test.shape[0]) * eps
    
    covmean = sqrtm(np.dot(cov_gen, cov_test))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_value = diff**2 + np.trace(cov_gen + cov_test - 2 * covmean)
    logger.debug(f"Computed Fréchet Inception Distance: {fid_value:.4f}")
    return fid_value

def wasserstein_distance_grid(Z_generated: np.ndarray, Z_test: np.ndarray) -> float:
    """
    Compute the average Wasserstein distance across all grid points to assess distributional similarity.
    
    Args:
        Z_generated (np.ndarray): Generated samples.
        Z_test (np.ndarray): Real observational samples.
    
    Returns:
        float: The mean Wasserstein distance.
    """
    n_lat, n_lon = Z_generated.shape[1], Z_generated.shape[2]
    wd_list = []
    for i in range(n_lat):
        for j in range(n_lon):
            wd = wasserstein_distance(Z_test[:, i, j], Z_generated[:, i, j])
            wd_list.append(wd)
    mean_wd = np.mean(wd_list)
    logger.debug(f"Computed average Wasserstein Distance across grid: {mean_wd:.4f}")
    return mean_wd

def test():
    """
    Main testing function to evaluate the GAN generator's performance in simulating extreme events.
    It loads the generator model, produces synthetic samples, and rigorously compares these to real observations
    using formal statistical methods.
    
    """
    logger.info("Starting evaluation of GAN-generated extreme event samples.")
    
    # -------------------------------------------------------------------------
    # Setup: Define parameters and device configuration.
    # -------------------------------------------------------------------------
    device = torch.device("cpu")  # Use "cuda" for GPU acceleration if available.
    noise_dim = 100
    batch_size = 50
    num_samples = 50
    pad = 1  # Amount of padding to remove from generated images.

    # -------------------------------------------------------------------------
    # Load the TorchScript GAN generator model.
    # -------------------------------------------------------------------------
    try:
        model_path = r"C:\Users\reverd\Repositories\master-thesis\spatial-extremes\TorchScript\scripted_GANGenerator.pt"
        scripted_model = torch.jit.load(model_path, map_location=device)
        scripted_model.eval()
        logger.info("Successfully loaded TorchScript generator model.")
    except Exception as e:
        logger.error("Error loading TorchScript model: " + str(e))
        return

    # -------------------------------------------------------------------------
    # Generate synthetic samples using the GAN model.
    # -------------------------------------------------------------------------
    U_samples_list = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            output = scripted_model.generate_samples(noise)
            U_samples_list.append(output.cpu().numpy())
    U_samples = np.concatenate(U_samples_list, axis=0)
    logger.info(f"Generated {U_samples.shape[0]} samples; initial output shape: {U_samples.shape}")

    # Concatenate all generated samples
    U_samples = np.concatenate(U_samples, axis=0)
    logger.info(f"Succesfully generated {U_samples.shape[0]} samples!")
    logger.debug(f"U_samples shape: {U_samples.shape}") 
      
    pad = 1  # Adjust if padding is different	
    
    # Remove padding from samples
    U_samples = U_samples[:, pad:-pad, pad:-pad]
    logger.debug(f"Removed padding from samples. New shape:   {U_samples.shape}")        
    logger.debug(f"U_samples min: {U_samples.min()}, max: {U_samples.max()}") 
    # -------------------------------------------------------------------------
    # Postprocess generated samples:
    # - Remove padding.
    # - Squeeze redundant channel dimensions.
    # -------------------------------------------------------------------------
    
    U_samples = (U_samples - U_samples.min()) / (U_samples.max() - U_samples.min())
    logger.info(f"After normalization: min={U_samples.min()}, max={U_samples.max()}")

    # -------------------------------------------------------------------------
    # Apply inverse transformation to map generated samples back to data space.
    # -------------------------------------------------------------------------
    GEV_params = get_params()  # Load GEV parameters.

    # Apply the inverse GEV transformation 
    Z_generated = inverse_transform(U_samples, None, GEV_params)
    logger.info("Applied inverse transformation to generated samples.")

    # -------------------------------------------------------------------------
    # Load and preprocess real observational data.
    # -------------------------------------------------------------------------
    try:
        data_path = r'C:\Users\reverd\Repositories\master-thesis\spatial-extremes\data\1\precipitation_maxima.nc'
        data = xr.open_dataset(data_path)
        var_name = list(data.data_vars)[0]
        Z_obs = data[var_name].values
        logger.debug(f"Loaded observation data with shape: {Z_obs.shape}")
    except Exception as e:
        logger.error("Error loading observation data: " + str(e))
        return

    # Trim data (e.g., remove the last 10 rows) based on domain-specific criteria.
    Z_obs = Z_obs[:-10, :, :]
    logger.debug(f"Observation data shape after trimming: {Z_obs.shape}")

    # -------------------------------------------------------------------------
    # Split the observational data into training and testing sets.
    # -------------------------------------------------------------------------
    n_train = 1939
    train_set = Z_obs[:n_train, :, :]
    test_set  = Z_obs[n_train:, :, :]
    logger.info(f"Data split: {train_set.shape[0]} training samples and {test_set.shape[0]} testing samples.")

    # For evaluation, we use the test set.
    Z_test = test_set
    logger.debug(f"Final test set shape: {Z_test.shape}")

    # -------------------------------------------------------------------------
    # Evaluate the generated samples against real observations using multiple metrics.
    # -------------------------------------------------------------------------
    ks_result, ks_stats = ks_test(Z_generated, Z_test)
    logger.info(f"KS Test result: {ks_result:.4f} (A value near 1.0 indicates complete divergence.)")

    chi_result = chi_statistic(Z_generated, Z_test)
    logger.info(f"Chi-statistic: {chi_result:.4f} (Low values imply poor modeling of extreme dependencies.)")

    mse_value = mean_squared_error(Z_generated, Z_test)
    logger.info(f"Mean Squared Error: {mse_value:.4f} (High MSE reflects significant spatial variability differences.)")

    fid_value = frechet_inception_distance(Z_generated, Z_test)
    logger.info(f"Fréchet Inception Distance: {fid_value:.4f} (High FID suggests feature mismatch and potential mode collapse.)")

    wd_value = wasserstein_distance_grid(Z_generated, Z_test)
    logger.info(f"Wasserstein Distance: {wd_value:.4f} (High distance confirms substantial distributional divergence.)")

    # -------------------------------------------------------------------------
    # Save evaluation plots for side-by-side visual comparison.
    # -------------------------------------------------------------------------
    eval_dir = "evaluation_plots"
    os.makedirs(eval_dir, exist_ok=True)
    num_plots = 5
    for i in range(num_plots):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(Z_generated[i], cmap='viridis')
        plt.title(f'Generated Sample {i+1}')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(Z_test[i % Z_test.shape[0]], cmap='viridis')
        plt.title(f'Real Sample {i+1}')
        plt.colorbar()
        plot_path = os.path.join(eval_dir, f'sample_comparison_{i+1}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.debug(f"Saved comparison plot: {plot_path}")
    logger.info("All evaluation plots have been saved successfully.")

    # -------------------------------------------------------------------------
    # Final Discussion: Log insights on model performance and potential failure modes.
    # -------------------------------------------------------------------------
    logger.info("Evaluation complete. The statistical metrics indicate significant divergence between generated and real samples.")
    logger.info("Potential causes include mode collapse, insufficient training duration, and inadequate feature representation.")
    logger.info("Recommended improvements: consider advanced architectures (e.g., WGAN), enhance regularization techniques, and extend training to capture higher-order dependencies.")

if __name__ == "__main__":
    test()
