# This pyPI script computes PI and associated analyses over the entire sample dataset
# which is from 2004, MERRA2.
#
# Created by Daniel Gilford, PhD (daniel.gilford@rutgers.edu)
# Many thanks to Daniel Rothenberg for his assitance optimizing pyPI
#
# Last updated 8/14/2020
#

# The input data needed to run this script (monthly data):
# 1. Sea surface temperature / Skin Temperature
# 2. Mean sea level pressure
# 3. Temperature (3D)
# 4. Specific humidity (3D)

# setup
import xarray as xr
import glob
import time
import pandas as pd
import numpy as np
import dask
import xesmf as xe
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# load in pyPI modules
from tcpyPI import pi
from tcpyPI.utilities import *


def run_PI(ds, dim='p', CKCD=0.9):
    """ This function calculates PI over the given dataset using xarray """
    
    # calculate PI over the whole data set using the xarray universal function
    result = xr.apply_ufunc(
        pi,
        ds['sst'], ds['psl'], ds[dim], ds['t'], ds['q'],
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


def search_df(df, **search):
    "search by keywords: match exactly or match substring"
    keys = ['activity_id', 'institution_id', 'source_id', 'experiment_id',
            'member_id', 'table_id', 'variable_id', 'grid_label']
    for skey in search.keys():
        if skey == 'variable_id':
            df = df[df[skey] == search[skey]]
        else:
            df = df[df[skey].str.contains(search[skey])]
    return df

def read_and_combine_netcdf(urls):
    # Use dask to parallelize the file reading
    dask.config.set(scheduler='threads')  # Use threads for local parallelization
    ds = xr.open_mfdataset(urls, concat_dim='time', combine='nested', decode_times=True, parallel=True, chunks={'time': 12})
    return ds

def time_series_check(ds, start_year, end_year):
    # Check time range & whether there's any year missing in between
    ds_time = ds['time']
    is_start_year_correct = (ds_time.min().dt.year.values == start_year)
    is_end_year_correct = (ds_time.max().dt.year.values == end_year)
    is_completed = (len(ds_time) == (end_year-start_year+1)*12)
    
    if is_completed and is_start_year_correct and is_end_year_correct:
        return True
    else:
        print("WARNING! Time Series do not meet the criteria.")
        return False

def load_and_check_variables(variable_id, params):
    # Unpack parameters from the dictionary
    df = params['df']
    experiment_id = params['experiment_id']
    table_id = params['table_id']
    member_id = params['member_id']
    source_id = params['source_id']
    start_year = params['start_year']
    end_year = params['end_year']

    my_df = search_df(df, experiment_id=experiment_id, table_id=table_id, 
                      variable_id=variable_id, member_id=member_id, source_id=source_id)

    if len(my_df) == 0:  # if there's no data
        print(f"NO DATA: {source_id} {member_id} {experiment_id} {variable_id}")
        return None

    url_list = my_df.URL.values
    result_ds = read_and_combine_netcdf(url_list).sel(time=slice(str(start_year), str(end_year)))
    if time_series_check(result_ds, start_year, end_year):
        print(f"# LOADED {source_id} {member_id} {experiment_id} {variable_id}")
        return result_ds
    else:
        return None


if __name__ == "__main__":

    # Get input datasets:
    df = pd.read_csv("http://mary.ldeo.columbia.edu/catalogs/cmip6_HighResMIP_opendap.csv")
    experiments = ['hist-1950', 'highresSST-present', 'highres-future', 'highresSST-future']
    names = ['highres_hm', 'highresSST_hm', 'highres_sm', 'highresSST_sm']
    # models = ['MPI-ESM1-2-XR', 'CNRM-CM6-1-HR', 
    #           'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM', 
    #           'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR', 
    #           'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR']
    # ens = ['r1i1p1f1', 'r1i1p1f2', 
    #        'r1i1p1f1', 'r1i2p1f1', 'r1i3p1f1',  
    #        'r1i1p2f1', 'r2i1p2f1', 'r3i1p2f1', 
    #        'r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']
    models = ['HadGEM3-GC31-HM', 'HadGEM3-GC31-HM',]
    ens = ['r1i2p1f1', 'r1i3p1f1', ]
    
    for iexp, iname in zip(experiments[3:4], names[3:4]):
        for imodel, iens in zip(models, ens):
            # get the range of year
            start_year, end_year = [1950, 2014] if iexp in experiments[:2] else [2015, 2050]
            if imodel=='EC-Earth3P-HR' and iens=='r1i1p1f1' and iexp=='highresSST-future':
                start_year, end_year = [2015, 2049]
            # get the dim name of level
            level_dim = 'lev' if imodel in ['MPI-ESM1-2-XR'] else 'plev'
            # pack all the params for this loop
            params = {'df': df, 'experiment_id': iexp, 'table_id': 'Amon', 'member_id': iens, 
                      'source_id': imodel, 'start_year': start_year, 'end_year':end_year}

            # Load data for input variables
            result_ds1 = load_and_check_variables('ts', params)
            result_ds2 = load_and_check_variables('psl', params)
            result_ds3 = load_and_check_variables('ta', params)
            result_ds4 = load_and_check_variables('hus', params)
            if any(result_ds is None for result_ds in [result_ds1, result_ds2, result_ds3, result_ds4]):
                continue

            # adjust units
            # target units: ts(sst)[C]; psl[hPa]; t[C] w/ Pa for altitude dimension; q(specific humidity)[g kg-1] w/ Pa for altitude dimension
            ts = result_ds1.ts - 273.15 if result_ds1.ts[0, 0, 0] > 100 else result_ds1.ts
            psl = result_ds2.psl / 100 if result_ds2.psl[0, 0, 0] > 9000 else result_ds2.psl
            t = result_ds3.ta.sortby(level_dim, ascending=False)
            t = t - 273.15 if t[level_dim][0].item() > 100 else t    
            t = t.assign_coords(**{level_dim: t[level_dim].data / 100}) if t[level_dim][0].item() > 9000 else t 
            q = result_ds4.hus.sortby(level_dim, ascending=False)
            q = q * 1000 if q[0,0,0,0] < 0.01 else q     
            q = q.assign_coords(**{level_dim: q[level_dim].data / 100}) if q[level_dim][0].item() > 9000 else q 
            
            # Unified time dimension
            psl = psl.assign_coords(time=ts['time'])
            t = t.assign_coords(time=ts['time'])
            q = q.assign_coords(time=ts['time'])
            if 'latitude' in ts.dims:
                ts = ts.rename({'latitude': 'lat', 'longitude': 'lon'})
                print("changed")
        
            # create new dataset that contains all the input variables
            ds = xr.Dataset(
                data_vars=dict(
                    sst = (['time','latitude','longitude'], ts.data), 
                    psl = (['time','latitude','longitude'], psl.data), 
                    t = (['time','p','latitude','longitude'], t.data), 
                    q = (['time','p','latitude','longitude'], q.data), 
                ),
                coords=dict(
                    p=("p", t[level_dim].data),
                    latitude=("latitude", ts.lat.data),
                    longitude=("longitude", ts.lon.data),
                    time=("time", ts.time.data),
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
            result.to_netcdf(f'/data0/yxia/TCGI/{imodel}/{iens}/PI/PI_{imodel}_{iens}_{iname}.nc', \
                              encoding=encoding_settings, mode="w")
            # result.to_netcdf('/home/yxia/data/try.nc', encoding=encoding_settings, mode="w")
            end_time2 = time.time()
            elapsed_time2 = end_time2 - start_time2
            print(f"Time taken on calculating and saving netCDF: {elapsed_time2:.2f} seconds")
            print('...PI computation complete and saved\n')
