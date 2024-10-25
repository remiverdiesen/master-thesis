import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import genextreme
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def fit_gev_to_grid_point(time_series):
    """
    Fit the GEV distribution to a single grid point time series.
    """
    if np.any(np.isnan(time_series)):
        return np.nan, np.nan, np.nan
    try:
        # Fit the GEV distribution to the time series using method='mm'
        shape, loc, scale = genextreme.fit(time_series, method='MLE')
        return shape, loc, scale
    except Exception as e:
        logger.warning(f"Could not fit GEV to time series: {e}")
        return np.nan, np.nan, np.nan

def fit_gev_margins(obs: np.ndarray, output_file_path: str):
    """
    Fit GEV distribution to each grid point of the input data in parallel.

    Parameters:
    - obs: A 3D numpy array with dimensions (time, lat, lon)
    - output_file_path: Path to the file where GEV parameters will be saved

    Returns:
    - GEV_params: A 3D numpy array of shape (lat, lon, 3) containing fitted GEV parameters
    """
    n_lat, n_lon = obs.shape[1], obs.shape[2]
    GEV_params = np.zeros((n_lat, n_lon, 3))

    # Flatten the grid indices for easier parallel processing
    indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]

    # Define a helper function to fit GEV for a single grid point and update results
    def process_grid_point(index):
        i, j = index
        time_series = obs[:, i, j]
        shape, loc, scale = fit_gev_to_grid_point(time_series)
        return (i, j, shape, loc, scale)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Wrap the executor.map call with tqdm for progress tracking
        results = list(tqdm(executor.map(process_grid_point, indices), total=len(indices), desc="Fitting GEV"))

    # Save results to the GEV_params array
    for i, j, shape, loc, scale in results:
        GEV_params[i, j, 0] = shape
        GEV_params[i, j, 1] = loc
        GEV_params[i, j, 2] = scale

    # Save GEV parameters to file
    with open(output_file_path, 'w') as file:
        for i, j, shape, loc, scale in results:
            if not np.isnan(shape):
                logger.debug(f"GEV fitted to grid point ({i}, {j}): shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
                file.write(f"({i+1}, {j+1}): {shape:.4f}, {loc:.4f}, {scale:.4f}\n")


####################################################################################

# precip_[Resolution]_[Time_Period]_[Season]_[Time_Interval].nc
 
EXPERIMENT = '2' 
DATASET = 'precip_2.4km_

dataset_file_path = f'spatial-extremes/data/{EXPERIMENT}/{DATASET}.nc'


output_file_path = f'spatial-extremes/experiments/{EXPERIMENT}/GEV_params.txt'


####################################################################################


ds = xr.open_dataset(dataset_file_path)  
var_name = list(ds.data_vars)[0]
Z_obs = ds[var_name].values

fit_gpd_margins(Z_obs, threshold, output_file_path)
