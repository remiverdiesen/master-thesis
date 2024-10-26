import os
import numpy as np
import xarray as xr
from scipy.stats import genpareto
from lightning.data import map
from tqdm import tqdm
import logging
import glob
from scipy.interpolate import griddata

import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the base logger level to DEBUG to capture all levels

# Create a handler for logging to a file with DEBUG level
file_handler = logging.FileHandler('egpd_fitting.log', mode='w')
file_handler.setLevel(logging.DEBUG)  # Logs everything to the file
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create a handler for logging to the console with INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only logs INFO level and above to the console
console_formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Remove any existing default handlers to prevent duplication
logger.propagate = False

import numpy as np
from scipy.optimize import minimize

def read_and_order_params(output_dir):
    """
    Read all parameter files, order them by grid point indices (i, j),
    and interpolate missing values with the nearest available parameter.
    
    Parameters:
    - output_dir: Directory containing parameter files.

    Returns:
    - ordered_params: A dictionary with interpolated (i, j) indices and their parameters.
    - original_params: A dictionary with the original (i, j) indices and their parameters.
    """
    # Initialize storage for parameters
    params_dict = {}

    # Collect all files matching the pattern
    txt_files = glob.glob(os.path.join(output_dir, "eGPD_params_worker_*.txt"))
    
    # Read all files and populate the dictionary
    for file_path in txt_files:
        with open(file_path, 'r') as infile:
            for line in infile:
                # Expected format: (i, j): shape, scale, loc, p
                parts = line.strip().split(':')
                if len(parts) != 2:
                    continue
                index_part, params_part = parts
                
                # Ensure index_part can be properly split into integers
                try:
                    index = tuple(int(x.strip()) for x in index_part.strip('()').split(','))
                except ValueError as e:
                    logger.warning(f"Failed to parse grid index from line: {line}. Error: {e}")
                    continue
                
                # Ensure params_part is properly formatted using list comprehension
                try:
                    params = [float(x.strip()) for x in params_part.split(',')]
                except ValueError as e:
                    logger.warning(f"Failed to parse parameters from line: {line}. Error: {e}")
                    continue

                # Store in dictionary
                params_dict[index] = params

        # Remove the individual worker file after processing
        os.remove(file_path)

    # Determine the maximum grid dimensions from the data
    max_i = max(idx[0] for idx in params_dict.keys())
    max_j = max(idx[1] for idx in params_dict.keys())

    # Create a grid for all indices
    full_grid_indices = [(i, j) for i in range(1, max_i + 1) for j in range(1, max_j + 1)]
    
    # Prepare data for interpolation
    known_indices = np.array(list(params_dict.keys()))
    known_values = np.array(list(params_dict.values()))
    
    # Interpolate missing values using nearest method
    interpolated_values = griddata(
        points=known_indices,  # Known points
        values=known_values,   # Known parameter values
        xi=full_grid_indices,  # All grid indices to fill
        method='nearest'       # Use nearest neighbor interpolation
    )

    # Create an ordered dictionary for all grid points
    ordered_params = {idx: params for idx, params in zip(full_grid_indices, interpolated_values)}
    
    # Return both the original parameters (without interpolation) and the ordered interpolated parameters
    return ordered_params, params_dict

    # Return both the original parameter

def save_ordered_params(output_dir, ordered_params, filename):
    """
    Save the ordered parameters to a file.
    
    Parameters:
    - output_dir: Directory to save the output file.
    - ordered_params: Ordered parameters.
    - filename: Name of the output file.
    """
    output_file_path = os.path.join(output_dir, filename)
    
    with open(output_file_path, 'w') as outfile:
        for (i, j), params in ordered_params.items():
            shape, scale, loc, p = params
            outfile.write(f"({i}, {j}): {shape:.4f}, {scale:.4f}, {loc:.4f}, {p:.4f}\n")


def neg_log_likelihood(params, exceedances):
    """
    Negative log-likelihood function for the EGPD.
    
    Parameters:
    - params: Array-like, parameters [xi, sigma, mu] to be optimized.
    - exceedances: Array of data points above the threshold.
    
    Returns:
    - Negative log-likelihood value.
    """
    xi, sigma, mu = params
    if sigma <= 0:
        return np.inf  # Scale parameter must be positive
    
    term = 1 + xi * (exceedances - mu) / sigma
    if np.any(term <= 0):  # Ensure valid domain
        return np.inf
    
    log_likelihood = -np.sum(np.log(sigma) + (-1 / xi) * np.log(term))
    return -log_likelihood

def fit_egpd_to_grid_point(time_series, threshold):
    """
    Fit the Extended GPD (eGPD) to a single grid point time series using a threshold.
    
    Parameters:
    - time_series: 1D array of values at a specific grid point.
    - threshold:   Threshold above which to fit the eGPD.
    
    Returns:
    - xi (shape), mu (location), sigma (scale), p: Fitted eGPD parameters.
    """
    # Data above the threshold (EGPD component)
    exceedances = time_series[time_series > threshold] - threshold

    # Probability of non-exceedance (values <= threshold)
    below_threshold = time_series[(time_series > 0) & (time_series <= threshold)]
    p = len(below_threshold) / len(time_series)

    if len(exceedances) < 1:
        return np.nan, np.nan, np.nan, p

    try:
        # Fit the GPD distribution to the exceedances using MLE
        shape, loc, scale = genpareto.fit(exceedances)
        # loc + threshold gives the estimate of mu
        mu_est = loc + threshold
        
        # Return the fitted parameters
        return shape, mu_est, scale, p
    except Exception as e:
        logger.warning(f"Could not fit eGPD to time series: {e}")
        return np.nan, np.nan, np.nan, p

def process_grid_subset(subset_indices, obs, threshold, worker_id, output_dir):
    """
    Process a subset of grid points, fit EGPD, and write results to a file.

    Parameters:
    - subset_indices: List of tuples (i, j) indicating grid indices to process
    - obs: A 3D numpy array with dimensions (time, lat, lon)
    - threshold: Threshold value for fitting EGPD
    - worker_id: Unique ID for the worker (used for file naming)
    - output_dir: Directory to save output files
    """
    # Use a shorter file name based on the worker ID
    output_filepath = os.path.join(output_dir, f"eGPD_params_worker_{worker_id[0]}.txt")
    
    with open(output_filepath, 'w') as file:
        for i, j in worker_id[1]:
            logger.debug(f"Processing grid point ({i}, {j})..")

            time_series = obs[:, i, j]
            shape, scale, loc, p = fit_egpd_to_grid_point(time_series, threshold)
            if np.isnan(shape):
                logger.debug(f"     failed for ({i}, {j}).")
            file.write(f"({i+1}, {j+1}): {shape:.4f}, {scale:.4f}, {loc:.4f}, {p:.4f}\n")
def main():
    # Load the dataset
    threshold = 0.01
    EXPERIMENT = '2'
    PERIOD = "2010-2024" # 
    SEASON = 'DJF'

    
    logger.info(f"\n\n\n Fitting eGPD params to grid points for Experiment {EXPERIMENT}, Period {PERIOD}, Season {SEASON}\n\n")

    # Use a wildcard to find the .nc file without specifying the full name
    dataset_file_path = glob.glob(f'spatial-extremes/data/{EXPERIMENT}/{PERIOD}/{SEASON}/*.nc')
    output_dir = f'spatial-extremes/experiments/{EXPERIMENT}/{PERIOD}/{SEASON}'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_dataset(dataset_file_path[0], engine='netcdf4')
    var_name = list(ds.data_vars)[0]
    Z_obs = ds[var_name].values

    # Determine the grid size
    n_lat, n_lon = Z_obs.shape[1], Z_obs.shape[2]
    indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]
    total_jobs = len(indices)
    logger.info(f"There are {len(ds.time)} time slices to process.")
    logger.info(f"Total of {total_jobs} grid points to process.")

    # Use os.cpu_count() to determine the number of workers (CPUs)
    num_workers = os.cpu_count()
    jobs_per_worker = total_jobs // num_workers
    logger.info(f"Using {num_workers} workers with {jobs_per_worker} pixels each.")

    job_splits = [indices[i:i + jobs_per_worker] for i in range(0, total_jobs, jobs_per_worker)]

    # Ensure each worker has roughly the same amount of work
    if len(job_splits) > num_workers:
        job_splits[-2].extend(job_splits.pop())

    # Define the function to process each subset
    def process_subset(worker_id, subset):
        process_grid_subset(subset, Z_obs, threshold, worker_id, output_dir)
    logger.info("\n\n\n     Starting the parallel processing of grid points....\n\n")

    # Use lightning's map function to distribute the work
    map(
        fn=process_subset,
        inputs=[(worker_id, job_splits[worker_id]) for worker_id in range(len(job_splits))],
        num_workers=num_workers,
        output_dir=output_dir
    )

    # Read, order, and save both the original and interpolated parameter files
    ordered_params, original_params = read_and_order_params(output_dir)

    # Save the original parameters (without interpolation)
    save_ordered_params(output_dir, original_params, "eGPD_params.txt")
    logger.info(f"All parameters have been saved into eGPD_params.txt")
    
    # Save the interpolated parameters
    save_ordered_params(output_dir, ordered_params, "eGPD_params_interpolated.txt")
    logger.info(f"All parameters have been interpolated eGPD_params_interpolated.txt.")


if __name__ == "__main__":
    main()
