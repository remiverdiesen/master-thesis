import os
import numpy as np
import xarray as xr
from scipy.stats import genpareto
from lightning.data import map
from tqdm import tqdm
import logging
import glob


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


import numpy as np
from scipy.optimize import minimize


def fit_egpd_to_grid_point(time_series, threshold):
    """
    Fit the Extended GPD (EGPD) to a single grid point time series using a threshold.
    
    Parameters:
    - time_series: 1D array of values at a specific grid point.
    - threshold: Threshold above which to fit the standard GPD part.
    
    Returns:
    - xi, sigma, mu, p: Fitted EGPD parameters.
    """
    # Non-zero data below threshold
    below_threshold = time_series[(time_series > 0) & (time_series <= threshold)]
    # Data above the threshold (standard GPD)
    exceedances = time_series[time_series > threshold] - threshold

    # Probability of non-exceedance (values <= threshold)
    p = len(below_threshold) / len(time_series)

    if len(exceedances) < 1:
        return np.nan, np.nan, np.nan, p

    try:
        # Fit the GPD distribution to the exceedances using MLE
        shape, loc, scale = genpareto.fit(exceedances)
        return shape, loc + threshold, scale, p
    except Exception as e:
        logger.warning(f"Could not fit EGPD to time series: {e}")
        return np.nan, np.nan, np.nan, p

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
    # Use a shorter file name based on the worker ID
    output_filepath = os.path.join(output_dir, f"eGPD_params_worker_{worker_id[0]}.txt")
    
    with open(output_filepath, 'w') as file:
        for i, j in worker_id[1]:
            time_series = obs[:, i, j]
            shape, loc, scale, p = fit_egpd_to_grid_point(time_series, threshold)
            if not np.isnan(shape):
                logger.debug(f"GPD fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}, p={p:.4f}")
                file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}, {p:.4f}\n")

def main():
    # Load the dataset
    threshold = 0.01
    EXPERIMENT = '2'
    PERIOD = "2010-2024" # 
    SEASON = 'DJF'

    
    logging.info(f"\n\n\n Fitting eGPD params to grid points for Experiment {EXPERIMENT}, Period {PERIOD}, Season {SEASON}\n\n")

    dataset_file_path = f'spatial-extremes/data/{EXPERIMENT}/{PERIOD}/{SEASON}/nc'
    output_dir = f'spatial-extremes/experiments/{EXPERIMENT}/{PERIOD}/{SEASON}'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ds = xr.open_dataset(dataset_file_path)
    var_name = list(ds.data_vars)[0]
    Z_obs = ds[var_name].values

    # Determine the grid size
    n_lat, n_lon = Z_obs.shape[1], Z_obs.shape[2]
    indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]
    total_jobs = len(indices)

    logging.info(f"Total of {total_jobs} grid points to process.")
    # Use os.cpu_count() to determine the number of workers (CPUs)
    num_workers = os.cpu_count()
    jobs_per_worker = total_jobs // num_workers
    logging.info(f"Using {num_workers} workers with {jobs_per_worker} jobs each.")

    job_splits = [indices[i:i + jobs_per_worker] for i in range(0, total_jobs, jobs_per_worker)]

    # Ensure each worker has roughly the same amount of work
    if len(job_splits) > num_workers:
        job_splits[-2].extend(job_splits.pop())


    # Define the function to process each subset
    def process_subset(worker_id, subset):
        process_grid_subset(subset, Z_obs, threshold, worker_id, output_dir)

    logging.info("\n\n\nStarting the parallel processing of grid points....\n\n\n")
    # Use lightning's map function to distribute the work
    map(
        fn=process_subset,
        inputs=[(worker_id, job_splits[worker_id]) for worker_id in range(len(job_splits))],
        num_workers=num_workers,
        output_dir=output_dir
    )

    # Read all params and order them in a new file and deleting the old ones
    all_params = []
     # Concatenate all the parameter files into one and delete the originals
    output_file_path = os.path.join(output_dir, "eGPD_params_all.txt")
    with open(output_file_path, 'w') as outfile:
        # Collect all files matching the pattern
        txt_files = glob.glob(os.path.join(output_dir, "eGPD_params_worker_*.txt"))
        for file_path in txt_files:
            with open(file_path, 'r') as infile:
                # Write each file's contents to the main output file
                outfile.write(infile.read())
            # Remove the individual worker file after processing
            os.remove(file_path)

    logging.info(f"All parameters have been consolidated into {output_file_path} and old files have been deleted.")

if __name__ == "__main__":
    main()
