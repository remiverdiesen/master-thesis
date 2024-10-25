import xarray as xr 

test_file = r'/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_1999-2010_Annual_5min.nc'
ds = xr.open_dataset(test_file)
var = 'Pr'
print("Read in dataset!")

# Define seasons for December-January-February, March-April-May, etc.
DJF = ds[var].sel(time=ds[var]['time.season'] == 'DJF')
print("DJF Done!")
MAM = ds[var].sel(time=ds[var]['time.season'] == 'MAM')
print("MAM Done!")
JJA = ds[var].sel(time=ds[var]['time.season'] == 'JJA')
print("JJA Done!")
SON = ds[var].sel(time=ds[var]['time.season'] == 'SON')
print("SON Done!")


# Save the results to separate files
DJF.to_netcdf('/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_1999-2010_DJF_5min.nc', engine='netcdf4', encoding={
    'lon': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'lat': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'i':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'j':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'Pr':  {'dtype': 'float32', 'zlib': True, 'complevel': 5}
})

MAM.to_netcdf('/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_1999-2010_MAM_5min.nc', engine='netcdf4', encoding={
    'lon': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'lat': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'i':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'j':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'Pr':  {'dtype': 'float32', 'zlib': True, 'complevel': 5}
})

JJA.to_netcdf('/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_1999-2010_JJA_5min.nc', engine='netcdf4', encoding={
    'lon': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'lat': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'i':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'j':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'Pr':  {'dtype': 'float32', 'zlib': True, 'complevel': 5}
})

SON.to_netcdf('/teamspace/studios/this_studio/spatial-extremes/data/2/precip_2.4km_1999-2010_SON_5min.nc', engine='netcdf4', encoding={
    'lon': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'lat': {'dtype': 'float32', 'zlib': True, 'complevel': 5},
    'i':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'j':   {'dtype': 'int16',   'zlib': True, 'complevel': 5}, 
    'Pr':  {'dtype': 'float32', 'zlib': True, 'complevel': 5}
})

