# TCGI calculation

1. **Start with single-year calculation checking**: "check_cal_ERA5.ipynb" runs one year of ERA5 data to check the calculation with parts of the plots. The completed output plots are available in "TCGI_singleyear_check_ERA5.pdf".
2. **Start calculate TCGI**: go to jupyter notebook "cal_TCGI.ipynb", which explains the calculation step by step.

## Calculation Process (ref. Camargo et al., 2014)
1. Calculate 5 variables:
    - Absolute Vorticity @ 850 hPa [s-1]
    - Vertical Wind Shear between 850- & 200-hPa [m s-1]
    - Column Relative Humidity
    - Saturation Deficit
    - Potential Intensity [m s-1] (based on Gilford, D. M. 2020 and Gilford, D. M. 2021)
2. For PI & CRH & SD, refill the land area. (explained in the following section, but we didn’t do this in the checking process)
3. Interpolate to 2*2 degrees resolution, and apply the land mask.
4. Get the annual climatology (year 1981-2010) differences between HighResMIP and ERA5 for each variables: $Correction_{var} = ERA5_{clim} – HighResMIP_{clim}$.
5. Add $Correction_{var}$ to 5 interpolated variables from HighResMIP.
6. Plug the coefficients from ERA5 (Calculated by Suzana) to get TCGI.

## Refill the land
PI, CRH, and SD are not meant to be used for land areas. When we change the resolution to a 2x2 degree grid, odd values can appear along the coast. To deal with this, we do the following:

At a certain latitude, we take the zonal mean climatology over the ocean region, which should be just one single value. Then fill the land at this latitude with this single value. We do this for every latitude, which helps smooth out the coastal area a bit.

Figures in "TCGI_singleyear_check_ERA5.pdf" show the differences between results with and without the refill-land process, demonstrating that our adjustments only affect coastal regions.
