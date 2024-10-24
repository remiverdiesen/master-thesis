import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import genpareto
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


##################################################################

EXPERIMENT = 1

if EXPERIMENT == 1:
    dataset = 'precipitation_maxima'
elif EXPERIMENT == 2:
    dataset = 'NL26_2km_5min'
elif EXPERIMENT == 3:
    dataset = 'NL26_10km_24h'


##################################################################


logger = logging.getLogger(__name__)

def fit_gpd_to_grid_point(time_series, threshold):
    """
    Fit the GPD distribution to a single grid point time series above a given threshold.
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

def fit_gpd_margins(obs: np.ndarray, threshold: float, output_file_path: str) -> np.ndarray:
    """
    Fit GPD distribution to each grid point of the input data in parallel.

    Parameters:
    - obs: A 3D numpy array with dimensions (time, lat, lon)
    - threshold: Threshold value for fitting GPD
    - output_file_path: Path to the file where GPD parameters will be saved

    Returns:
    - GPD_params: A 3D numpy array of shape (lat, lon, 3) containing fitted GPD parameters
    """
    n_lat, n_lon = obs.shape[1], obs.shape[2]
    GPD_params = np.zeros((n_lat, n_lon, 3))

    # Flatten the grid indices for easier parallel processing
    indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]

    # Define a helper function to fit GPD for a single grid point and update results
    def process_grid_point(index):
        i, j = index
        time_series = obs[:, i, j]
        shape, loc, scale = fit_gpd_to_grid_point(time_series, threshold)
        return (i, j, shape, loc, scale)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Wrap the executor.map call with tqdm for progress tracking
        results = list(tqdm(executor.map(process_grid_point, indices), total=len(indices), desc="Fitting GPD"))

    # Save results to the GPD_params array
    for i, j, shape, loc, scale in results:
        GPD_params[i, j, 0] = shape
        GPD_params[i, j, 1] = loc
        GPD_params[i, j, 2] = scale

    # Save GPD parameters to file
    with open(output_file_path, 'w') as file:
        for i, j, shape, loc, scale in results:
            if not np.isnan(shape):
                print(f"GPD fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
                file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")


# Define a threshold for GPD fitting (adjust based on your analysis needs)
threshold = 0.01  # Example threshold value

fp = f'.\data\{EXPERIMENT}\{dataset}.nc'
ds = xr.open_dataset(fp)
var_name = list(ds.data_vars)[0]
Z_obs = ds[var_name].values

file_path = f'.\experiments\{EXPERIMENT}\gpd_params.txt'
fit_gpd_margins(Z_obs, threshold, file_path)