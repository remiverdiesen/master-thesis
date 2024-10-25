import os
import numpy as np
import xarray as xr
from scipy.stats import genpareto
from lightning.data import map
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def fit_gpd_to_grid_point(time_series, threshold):
    """
    Fit the GPD distribution to a single grid point time series using a threshold.
    """
    # Filter data above the threshold
    exceedances = time_series[time_series > threshold] - threshold
    
    if len(exceedances) < 1:
        return np.nan, np.nan, np.nan
    
    try:
        # Fit the GPD distribution to the exceedances using MLE
        shape, loc, scale = genpareto.fit(exceedances)
        return shape, loc + threshold, scale
    except Exception as e:
        logger.warning(f"Could not fit GPD to time series: {e}")
        return np.nan, np.nan, np.nan

def process_grid_subset(subset_indices, obs, threshold, worker_id, output_dir):
    """
    Process a subset of grid points, fit GPD, and write results to a file.

    Parameters:
    - subset_indices: List of tuples (i, j) indicating grid indices to process
    - obs: A 3D numpy array with dimensions (time, lat, lon)
    - threshold: Threshold value for fitting GPD
    - worker_id: Unique ID for the worker (used for file naming)
    - output_dir: Directory to save output files
    """
    output_filepath = os.path.join(output_dir, f"GPD_params_worker_{worker_id}.txt")

    with open(output_filepath, 'w') as file:
        for i, j in subset_indices:
            time_series = obs[:, i, j]
            shape, loc, scale = fit_gpd_to_grid_point(time_series, threshold)
            if not np.isnan(shape):
                logger.debug(f"GPD fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
                file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")

def main():
    # Load the dataset
    threshold = 0.01
    EXPERIMENT = '2'
    DATASET = 'precip_2.4km'
    dataset_file_path = f'spatial-extremes/data/{EXPERIMENT}/{DATASET}.nc'
    output_dir = f'spatial-extremes/experiments/{EXPERIMENT}'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ds = xr.open_dataset(dataset_file_path)
    var_name = list(ds.data_vars)[0]
    Z_obs = ds[var_name].values

    # Determine the grid size
    n_lat, n_lon = Z_obs.shape[1], Z_obs.shape[2]
    indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]
    total_jobs = len(indices)

    # Use os.cpu_count() to determine the number of workers (CPUs)
    num_workers = os.cpu_count()
    jobs_per_worker = total_jobs // num_workers
    job_splits = [indices[i:i + jobs_per_worker] for i in range(0, total_jobs, jobs_per_worker)]

    # Ensure each worker has roughly the same amount of work
    if len(job_splits) > num_workers:
        job_splits[-2].extend(job_splits.pop())

    # Define the function to process each subset
    def process_subset(worker_id, subset):
        process_grid_subset(subset, Z_obs, threshold, worker_id, output_dir)

    # Use lightning's map function to distribute the work
    map(
        fn=process_subset,
        inputs=[(worker_id, job_splits[worker_id]) for worker_id in range(len(job_splits))],
        num_workers=num_workers,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()


# import numpy as np
# import xarray as xr
# from concurrent.futures import ThreadPoolExecutor
# from scipy.stats import genpareto
# from tqdm import tqdm
# import logging
# from concurrent.futures import ProcessPoolExecutor


# logger = logging.getLogger(__name__)

# def fit_gpd_to_grid_point(time_series, threshold):
#     # Filter data above the threshold
#     exceedances = time_series[time_series > threshold] - threshold
    
#     if len(exceedances) < 1:
#         return np.nan, np.nan, np.nan
    
#     try:
#         # Fit the GPD distribution to the exceedances using MLE
#         shape, loc, scale = genpareto.fit(exceedances)
#         return shape, loc + threshold, scale
#     except Exception as e:
#         logger.warning(f"Could not fit GPD to time series: {e}")
#         return np.nan, np.nan, np.nan

# # Move the helper function to a global scope
# def process_grid_point(args):
#     obs, threshold, i, j = args
#     time_series = obs[:, i, j]
#     shape, loc, scale = fit_gpd_to_grid_point(time_series, threshold)
#     return (i, j, shape, loc, scale)

# def fit_gpd_margins(obs: np.ndarray, threshold: float, output_file_path: str):
#     """
#     Fit GPD distribution to each grid point of the input data in parallel.

#     Parameters:
#     - obs:              A 3D numpy array with dimensions (time, lat, lon)
#     - threshold:        Threshold value for fitting GPD
#     - output_file_path: Path to the file where GPD parameters will be saved

#     Returns:
#     - GPD_params: A 3D numpy array of shape (lat, lon, 3) containing fitted GPD parameters
#     """
#     n_lat, n_lon = obs.shape[1], obs.shape[2]
#     GPD_params = np.zeros((n_lat, n_lon, 3))

#     # Flatten the grid indices for easier parallel processing
#     indices = [(obs, threshold, i, j) for i in range(n_lat) for j in range(n_lon)]

#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor(max_workers=32) as executor:
#         # Wrap the executor.map call with tqdm for progress tracking
#         results = list(tqdm(executor.map(process_grid_point, indices), total=len(indices), desc="Fitting GPD"))

#     # Save results to the GPD_params array
#     for i, j, shape, loc, scale in results:
#         GPD_params[i, j, 0] = shape
#         GPD_params[i, j, 1] = loc
#         GPD_params[i, j, 2] = scale

#     # Save GPD parameters to file
#     with open(output_file_path, 'w') as file:
#         for i, j, shape, loc, scale in results:
#             if not np.isnan(shape):
#                 print(f"GPD fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
#                 file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")


# ####################################################################################

# # precip_[Resolution]_[Time_Period]_[Season]_[Time_Interval].nc


# threshold = 0.01  
# EXPERIMENT = '2' 
# DATASET = 'precip_2.4km_

# dataset_file_path = f'spatial-extremes/data/{EXPERIMENT}/{DATASET}.nc'


# output_file_path = f'spatial-extremes/experiments/{EXPERIMENT}/GPD_params.txt'


# ####################################################################################


# ds = xr.open_dataset(dataset_file_path)  
# var_name = list(ds.data_vars)[0]
# Z_obs = ds[var_name].values

# fit_gpd_margins(Z_obs, threshold, output_file_path)