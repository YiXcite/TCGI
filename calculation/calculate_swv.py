import xarray as xr
import glob
import time
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
import dask
import xesmf as xe
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import functions


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

df = pd.read_csv("http://mary.ldeo.columbia.edu/catalogs/cmip6_HighResMIP_opendap.csv")
experiments = ['hist-1950', 'highresSST-present', 'highres-future', 'highresSST-future']
names = ['highres_hm', 'highresSST_hm', 'highres_sm', 'highresSST_sm']
models = ['MPI-ESM1-2-XR', 'CNRM-CM6-1-HR',
          'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM',
          'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR', ]
ens = ['r1i1p1f1', 'r1i1p1f2',
       'r1i1p1f1', 'r1i2p1f1', 'r1i3p1f1',
       'r1i1p2f1', 'r2i1p2f1', 'r1i1p1f1', 'r2i1p1f1', ]

cal_var = 'saturated_water_vapor'  # function and input in the following code need to be changed
cal_var_abbr = 'swv'

for iexp, iname in zip(experiments, names):
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
        result_ds1 = load_and_check_variables('ta', params)
        result_ds2 = load_and_check_variables('psl', params)
        result_ds3 = load_and_check_variables('ts', params)
        if any(result_ds is None for result_ds in [result_ds1, result_ds2, result_ds3]):
            continue
            
        start_time2 = time.time()
        result = functions.saturated_water_vapor(result_ds1.ta, result_ds2.psl, result_ds3.ts, level_dim)
        result.to_netcdf(f'/data0/yxia/TCGI/{imodel}/{iens}/{cal_var_abbr}/{cal_var_abbr}_{imodel}_{iens}_{iname}.nc', \
                          encoding={f'{cal_var_abbr}': {'zlib': True, 'complevel': 5}}, mode="w")
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"Time taken on calculating and saving netCDF: {elapsed_time2:.2f} seconds")
        print(f"# GET {imodel} {iens} {iname} {cal_var}\n")
