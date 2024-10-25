import os
from tqdm import tqdm
import xarray as xr

# Step 1: Set directory path
directory = r'/teamspace/studios/this_studio/spatial-extremes/data/2'

# Step 2: Get list of all files in the 'year/month' subdirectories
files = []
for year in os.listdir(directory)[14:]:
    year_path = os.path.join(directory, year)
    if os.path.isdir(year_path):
        for month in os.listdir(year_path):
            month_path = os.path.join(year_path, month)
            if os.path.isfile(month_path) and month_path.endswith('.nc'):
                files.append(month_path)

# Step 3: Sort the files to ensure chronological order
files.sort()  # Ensure files are in chronological order

# Step 4: Use xarray to open and concatenate the NetCDF files
# Using Dask for parallelized operations and optimized chunking
combined_ds = xr.open_mfdataset(
    files,
    concat_dim='time',
    combine='nested',
    engine='netcdf4'
)

# Step 5: Save the concatenated dataset to a new NetCDF file with optimal settings
output_path = r'/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_2010-2020_Annual_5min_2.nc'
combined_ds.to_netcdf(
    output_path,
    engine='netcdf4',
    encoding={
        'lon': {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 7,  # Higher compression for disk efficiency
            'shuffle': True

        },
        'lat': {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 7,
            'shuffle': True

        },
        'i': {
            'dtype': 'int16',
            'zlib': True,
            'complevel': 7,
            'shuffle': True

        },
        'j': {
            'dtype': 'int16',
            'zlib': True,
            'complevel': 7,
            'shuffle': True

        },
        'Pr': {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 7,
            'shuffle': True
            # Dynamically calculated chunk sizes for Pr

        }
    }
)

print("Files successfully concatenated into precip_2.4km_2010-2024_Annual_1hour.nc")

