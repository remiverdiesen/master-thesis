import xarray as xr 

test_file = r'/teamspace/studios/this_studio/spatial-extremes/data/2/NL16_2_4km_5min.nc'
ds = xr.open_dataset(test_file)
var = 'Pr'
print("Read in dataset!")
hourly_sum = ds[var].resample(time='1h').sum()
print("Hourly sum Done!")
# Define seasons for December-January-February, March-April-May, etc.
DJF = hourly_sum.sel(time=hourly_sum['time.season'] == 'DJF')
print("DJF Done!")
MAM = hourly_sum.sel(time=hourly_sum['time.season'] == 'MAM')
print("MAM Done!")
JJA = hourly_sum.sel(time=hourly_sum['time.season'] == 'JJA')
print("JJA Done!")
SON = hourly_sum.sel(time=hourly_sum['time.season'] == 'SON')
print("SON Done!")
# Define annual data
annual = hourly_sum

# Save the results to separate files
DJF.to_netcdf("NL12_2_4km_1h_DJF.nc", engine='netcdf', encoding={
    'time': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})

MAM.to_netcdf("NL12_2_4km_1h_MAM.nc", engine='netcdf', encoding={
    'time': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})

JJA.to_netcdf("NL12_2_4km_1h_JJA.nc", engine='netcdf', encoding={
    'time': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})

SON.to_netcdf("NL12_2_4km_1h_SON.nc", engine='netcdf', encoding={
    'time': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})

annual.to_netcdf("NL12_2_4km_1h.nc", engine='netcdf', encoding={
    'time': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True},
    'Pr': {'dtype': 'float32', 'zlib': True, 'complevel': 5, 'shuffle': True}
})