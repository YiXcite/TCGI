# This pyPI script computes PI and associated analyses over the entire sample dataset
# which is from 2004, MERRA2.
#
# Created by Daniel Gilford, PhD (daniel.gilford@rutgers.edu)
# Many thanks to Daniel Rothenberg for his assitance optimizing pyPI
#
# Last updated 8/14/2020
#

# The input data needed to run this script (ERA5 monthly data in the year 2020):
# 1. Sea surface temperature
# 2. Mean sea level pressure
# 3. Temperature (3D)
# 4. Specific humidity (3D)

# setup
import xarray as xr
import pickle
import xesmf as xe
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# load in pyPI modules
from tcpyPI import pi
from tcpyPI.utilities import *


# define the sample data locations
datdir='/data0/yxia/TCGI/'
# _FN=datdir+'sample_data.nc'
# _mdrF=datdir+'mdr.pk1' 
    

def run_PI(ds, dim='p',CKCD=0.9):
    """ This function calculates PI over the given dataset using xarray """
    
    # calculate PI over the whole data set using the xarray universal function
    result = xr.apply_ufunc(
        pi,
        ds['sst'], ds['msl'], ds[dim], ds['t'], ds['q'],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1.0, V_reduc=0.8, ptop=30, miss_handle=1),
        input_core_dims=[[], [], [dim, ], [dim, ], [dim, ],],
        output_core_dims=[[], [], [], [], []],
        vectorize=True   # Vectorize the operation for better performance
    )

    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result
    vmax = vmax.fillna(0)
    
    out_ds=xr.Dataset({
        'vmax': vmax, 
        'pmin': pmin,
        'ifl': ifl,
        't0': t0,
        'otl': otl,
        })
    
    # add names and units to the structure
    out_ds.vmax.attrs['standard_name'],out_ds.vmax.attrs['units']='Maximum Potential Intensity','m/s'
    out_ds.pmin.attrs['standard_name'],out_ds.pmin.attrs['units']='Minimum Central Pressure','hPa'
    out_ds.ifl.attrs['standard_name']='pyPI Flag'
    out_ds.t0.attrs['standard_name'],out_ds.t0.attrs['units']='Outflow Temperature','K'
    out_ds.otl.attrs['standard_name'],out_ds.otl.attrs['units']='Outflow Temperature Level','hPa'

    # return the output from pi.py as an xarray data structure
    return out_ds  
    

if __name__ == "__main__":

    # Get input datasets:
    era5_path = "/xpt/berimbau.local/data1/suzana/ERA5/OriginalFiles2020/"
    sst = xr.open_dataset(era5_path + "SST_2020_monthly.nc").sst.sortby('latitude') - 273.15
    msl = xr.open_dataset(era5_path + "MSLP_2020_monthly.nc").msl.sortby('latitude') / 100
    t = xr.open_dataset(era5_path + "T_2020_monthly.nc").t.sortby('latitude').sortby('level', ascending=False) - 273.15
    q = xr.open_dataset(era5_path + "Q_2020_monthly.nc").q.sortby('latitude').sortby('level', ascending=False) * 1000
    
    # Unified time dimension
    msl = msl.assign_coords(time=sst['time'])
    t = t.assign_coords(time=sst['time'])
    q = q.assign_coords(time=sst['time'])

    # create new dataset that contains all the input variables
    ds = xr.Dataset(
        data_vars=dict(
            sst = (['time','latitude','longitude'], sst.data), 
            msl = (['time','latitude','longitude'], msl.data), 
            t = (['time','p','latitude','longitude'], t.data), 
            q = (['time','p','latitude','longitude'], q.data), 
        ),
        coords=dict(
            p=("p", t.level.data),
            latitude=("latitude", sst.latitude.data),
            longitude=("longitude", sst.longitude.data),
            time=("time", sst.time.data),
        )
    )

    # Execute PI analysis over the whole dataset and save the output
    start_time2 = time.time()
    print('Beginning PI computations...')
    ds = ds.load()
    result = run_PI(ds)

    encoding_settings = {
                'vmax': {'zlib': True, 'complevel': 5},
                'pmin': {'zlib': True, 'complevel': 5},
                'ifl': {'zlib': True, 'complevel': 5},
                't0': {'zlib': True, 'complevel': 5},
                'otl': {'zlib': True, 'complevel': 5},
            }
    result = result.rename({'latitude': 'lat', 'longitude':'lon'})
    result.to_netcdf('/home/yxia/data/PI_ERA5_2020_2.nc', encoding=encoding_settings, mode="w")
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print(f"Time taken on calculating and saving netCDF: {elapsed_time2:.2f} seconds")
    print('...PI computation complete and saved\n')