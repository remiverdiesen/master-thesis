import os
import glob
import logging

import numpy as np
import xarray as xr


from scipy.stats import genextreme
from lightning.data import map

from scipy.spatial import cKDTree


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the base logger level to DEBUG to capture all levels

# Create a handler for logging to a file with DEBUG level
file_handler = logging.FileHandler('gev_fitting.log', mode='w')
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

def fill_nan_with_nearest(known_indices, known_values, full_grid_indices):
    # Remove rows with NaN in known_values for interpolation
    valid_mask = ~np.isnan(known_values).any(axis=1)
    valid_indices = known_indices[valid_mask]
    valid_values = known_values[valid_mask]

    # Build a KDTree for efficient nearest-neighbor search
    tree = cKDTree(valid_indices)

    # Find the nearest valid point for each grid index
    _, nearest_idx = tree.query(full_grid_indices)

    # Fill the NaN values with the nearest non-NaN data
    filled_values = valid_values[nearest_idx]
    
    return filled_values

def read_and_order_params(output_dir):
    # Initialize storage for parameters
    params_dict = {}

    # Collect all files matching the pattern
    txt_files = glob.glob(os.path.join(output_dir, "GEV_params_worker_*.txt"))
    
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
    
    # Use KDTree-based method to fill NaN values with the nearest available value
    interpolated_values = fill_nan_with_nearest(known_indices, known_values, full_grid_indices)

    # Create an ordered dictionary for all grid points
    ordered_params = {idx: params for idx, params in zip(full_grid_indices, interpolated_values)}
    
    # Return both the original parameters (without interpolation) and the ordered interpolated parameters
    return ordered_params, params_dict

def save_ordered_params(output_dir, ordered_params, filename):

    output_file_path = os.path.join(output_dir, filename)
    
    with open(output_file_path, 'w') as outfile:
        for (i, j), params in ordered_params.items():
            shape, scale, loc = params
            outfile.write(f"({i}, {j}): {shape:.4f}, {scale:.4f}, {loc:.4f}\n")
            
def fit_gev_to_grid_point(time_series):
    """
    Fit the GEV distribution to a single grid point time series.
    """
    if np.any(np.isnan(time_series)):
        return np.nan, np.nan, np.nan
    try:
        # Fit the GEV distribution to the time series using method='MLE'
        shape, loc, scale = genextreme.fit(time_series, method='MLE')
        return shape, loc, scale
    except Exception as e:
        logger.warning(f"Could not fit GEV to time series: {e}")
        return np.nan, np.nan, np.nan

def process_grid_subset(subset_indices, obs, worker_id, output_dir):
    """
    Process a subset of grid points, fit GEV, and write results to a file.

    Parameters:
    - subset_indices: List of tuples (i, j) indicating grid indices to process
    - obs: A 3D numpy array with dimensions (time, lat, lon)
    - worker_id: Unique ID for the worker (used for file naming)
    - output_dir: Directory to save output files
    """
    output_filepath = os.path.join(output_dir, f"GEV_params_worker_{worker_id[0]}.txt")

    with open(output_filepath, 'w') as file:
        for i, j in worker_id[1]:
            time_series = obs[:, i, j]
            shape, loc, scale = fit_gev_to_grid_point(time_series)
            if not np.isnan(shape):
                logger.debug(f"GEV fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
                file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")

def main():
    # Load the dataset

    EXPERIMENT = '4'
    PERIOD     = "1998-2010" #  "1998-2010" "2010-2024"
    SEASON     = 'DJF'         # 

    
    logger.info(f"\n\n\n Fitting GPD params to grid points for Experiment {EXPERIMENT}, Period {PERIOD}, Season {SEASON}\n\n")

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
        process_grid_subset(subset, Z_obs, worker_id, output_dir)
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
    save_ordered_params(output_dir, original_params, "GEV_params.txt")
    logger.info(f"All parameters have been saved into GEV_params.txt")
    
    # Save the interpolated parameters
    save_ordered_params(output_dir, ordered_params, "GEV_params_interpolated.txt")
    logger.info(f"All parameters have been interpolated GEV_params_interpolated.txt.")

if __name__ == "__main__":
    main()



# import numpy as np
# import xarray as xr
# from concurrent.futures import ThreadPoolExecutor
# from scipy.stats import genextreme
# from tqdm import tqdm
# import logging

# logger = logging.getLogger(__name__)


# def fit_gev_to_grid_point(time_series):
#     """
#     Fit the GEV distribution to a single grid point time series.
#     """
#     if np.any(np.isnan(time_series)):
#         return np.nan, np.nan, np.nan
#     try:
#         # Fit the GEV distribution to the time series using method='mm'
#         shape, loc, scale = genextreme.fit(time_series, method='MLE')
#         return shape, loc, scale
#     except Exception as e:
#         logger.warning(f"Could not fit GEV to time series: {e}")
#         return np.nan, np.nan, np.nan

# def fit_gev_margins(obs: np.ndarray, output_file_path: str):
#     """
#     Fit GEV distribution to each grid point of the input data in parallel.

#     Parameters:
#     - obs: A 3D numpy array with dimensions (time, lat, lon)
#     - output_file_path: Path to the file where GEV parameters will be saved

#     Returns:
#     - GEV_params: A 3D numpy array of shape (lat, lon, 3) containing fitted GEV parameters
#     """
#     n_lat, n_lon = obs.shape[1], obs.shape[2]
#     GEV_params = np.zeros((n_lat, n_lon, 3))

#     # Flatten the grid indices for easier parallel processing
#     indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]

#     # Define a helper function to fit GEV for a single grid point and update results
#     def process_grid_point(index):
#         i, j = index
#         time_series = obs[:, i, j]
#         shape, loc, scale = fit_gev_to_grid_point(time_series)
#         return (i, j, shape, loc, scale)

#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor() as executor:
#         # Wrap the executor.map call with tqdm for progress tracking
#         results = list(tqdm(executor.map(process_grid_point, indices), total=len(indices), desc="Fitting GEV"))

#     # Save results to the GEV_params array
#     for i, j, shape, loc, scale in results:
#         GEV_params[i, j, 0] = shape
#         GEV_params[i, j, 1] = loc
#         GEV_params[i, j, 2] = scale

#     # Save GEV parameters to file
#     with open(output_file_path, 'w') as file:
#         for i, j, shape, loc, scale in results:
#             if not np.isnan(shape):
#                 logger.debug(f"GEV fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
#                 file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")


# ####################################################################################

# # precip_[Resolution]_[Time_Period]_[Season]_[Time_Interval].nc
 
# EXPERIMENT = '2' 
# DATASET = 'precip_2.4km_

# dataset_file_path = f'spatial-extremes/data/{EXPERIMENT}/{DATASET}.nc'


# output_file_path = f'spatial-extremes/experiments/{EXPERIMENT}/GEV_params.txt'


# ####################################################################################


# ds = xr.open_dataset(dataset_file_path)  
# var_name = list(ds.data_vars)[0]
# Z_obs = ds[var_name].values

# fit_gpd_margins(Z_obs, threshold, output_file_path)
