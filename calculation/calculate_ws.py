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


df = pd.read_csv("http://mary.ldeo.columbia.edu/catalogs/cmip6_HighResMIP_opendap.csv")
experiments = ['hist-1950', 'highresSST-present', 'highres-future', 'highresSST-future']
names = ['highres_hm', 'highresSST_hm', 'highres_sm', 'highresSST_sm']
models = ['MPI-ESM1-2-XR', 'CNRM-CM6-1-HR',
          'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-HM',
          'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR',
          'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR']
ens = ['r1i1p1f1', 'r1i1p1f2',
       'r1i1p1f1', 'r1i2p1f1', 'r1i3p1f1',
       'r1i1p2f1', 'r2i1p2f1', 'r3i1p2f1',
       'r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']

cal_var = 'wind_shear'  # function and input in the following code need to be changed
cal_var_abbr = 'ws'

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
        result_ds1 = functions.load_and_check_variables('ua', params)
        result_ds2 = functions.load_and_check_variables('va', params)
        if any(result_ds is None for result_ds in [result_ds1, result_ds2]):
            continue
            
        start_time2 = time.time()
        result = functions.wind_shear(result_ds1.ua, result_ds2.va, level_dim)            
        result.to_netcdf(f'/data0/yxia/TCGI/{imodel}/{iens}/{cal_var_abbr}/{cal_var_abbr}_{imodel}_{iens}_{iname}.nc', \
                          encoding={f'{cal_var_abbr}': {'zlib': True, 'complevel': 5}}, mode="w")
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"Time taken on calculating and saving netCDF: {elapsed_time2:.2f} seconds")
        print(f"# GET {imodel} {iens} {iname} {cal_var}\n")
