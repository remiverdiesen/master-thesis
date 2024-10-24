import xarray as xr
import os

# Step 1: Set directory path
directory = r'/teamspace/studios/this_studio/spatial-extremes/data/2'

# Step 2: Get list of all files in the 'year/month' subdirectories
files = []
for year in os.listdir(directory):
    # print(year)
    year_path = os.path.join(directory, year)
    if os.path.isdir(year_path):
        for month in os.listdir(year_path):
            month_path = os.path.join(year_path, month)
            print(month_path)
            if os.path.isfile(month_path) and month_path.endswith('.nc'):
                files.append(month_path)

# Step 3: Sort the files to ensure chronological order
files.sort()  # Sort to ensure files are in chronological order

# # Step 4: Use xarray to open and concatenate the NetCDF files
# combined_ds_concat = xr.concat([xr.open_dataset(f) for f in files], dim='time')

# # Sort the time index to ensure it's monotonic
# combined_ds_concat = combined_ds_concat.sortby('time')

from tqdm import tqdm
import xarray as xr

# Step 4: Use tqdm to show progress of opening datasets
datasets = []
for f in tqdm(files, desc="Opening NetCDF files"):
    datasets.append(xr.open_dataset(f))

# Concatenate with tqdm tracking
combined_ds_concat = xr.concat(tqdm(datasets, desc="Concatenating datasets"), dim='time')

# Sort the time index to ensure it's monotonic
combined_ds_concat = combined_ds_concat.sortby('time')


# # Step 5: Save the concatenated dataset to a new NetCDF file
output_path = r'/teamspace/studios/this_studio/spatial-extremes/data/2/NL16_2_4km_1999_2010.nc'
# combined_ds.to_netcdf(output_path)
combined_ds_concat.to_netcdf(output_path, engine='h5netcdf', encoding={
    'lon': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'lat': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'i':   {'dtype': 'int16',   'zlib': True, 'complevel': 5, 'shuffle': True},
    'j':   {'dtype': 'int16',   'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr':  {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})
print("Files successfully concatenated into combined_rainfall_data.nc")
