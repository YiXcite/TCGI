# TCGI: Tropical Cyclone Genesis Index

## Instructions

1. **Single-Year Calculation Check.** Execute "check_cal_ERA5.ipynb" to validate the calculation with one year of ERA5 data.
   - Resulting plots are available in "TCGI_singleyear_check_ERA5.pdf".
   - Output data is available in the folder "output_ERA5_2020_2deg".

2. **Start TCGI Calculation.** Navigate to the "cal_TCGI.ipynb" for a step-by-step explanation of the TCGI calculation.

Required Monthly ERA5 Input Data:
- U-component of wind at 850hPa & 200hPa
- V-component of wind at 850hPa & 200hPa
- Temperature (3D)
- Mean sea level pressure
- Skin temperature
- Total column water vapor
- Sea Surface Temperature
- Specific humidity (3D)

## Calculation Process (Camargo et al., 2014)
1. Computation of five critical variables:
    - Clipped Absolute Vorticity at 850 hPa $[10^{-5} \cdot s^{-1}]$: minimum of $(\eta)$ and $(3.7 \times 10^{-5})$.
    - Vertical Wind Shear between 850 and 200 hPa $[m \cdot s^{-1}]$.
    - Column Relative Humidity [\%]: column-integrated water vapor / saturated water vapor.
    - Saturation Deficit: saturated water vapor minus column-integrated water vapor.
    - Potential Intensity $[m \cdot s^{-1}]$ following Gilford, D. M. 2020 and Gilford, D. M. 2021.

2. Refill land areas for Potential Intensity, Column Relative Humidity, and Saturation Deficit after interpolation. (Explained in the next section)
3. Interpolate data to a $2^\circ \times 2^\circ$ grid resolution and apply the land mask.
4. Derive annual climatological differences between HighResMIP and ERA5 for each variable:
   - $Correction_{var} = ERA5_{clim} - HighResMIP_{clim}\$.
5. Add $Correction_{var}$ to 5 interpolated variables from HighResMIP.
6. Insert coefficients derived from ERA5 (calculated by Suzana Camargo) to calculate TCGI:
   - $TCGI = \exp(b+b_H H+b_T T+b_{\eta} \eta+b_V V+\log(\cos\phi))$, where $b_X$ represents the coefficients; $H$ is the humidity component (CRH/SD); $T$ is potential intensity; $V$ is vertical shear; $\eta$ is clipped absolute vorticity; and $\phi$ is latitude.

**Note**: The use of ERA5-derived coefficients necessitates steps 4 & 5 to align model outputs with ERA5 benchmarks.

## Land Refilling Process
PI, CRH, and SD are not meant to be used for land areas. When we change the resolution to a $2^\circ \times 2^\circ$ grid, odd values can appear along the coast. To deal with this, we do the following:

At a certain latitude, we take the zonal mean climatology over the ocean region, which should be just one single value. Then fill the land at this latitude with this single value. We do this for every latitude, which helps smooth out the coastal area a bit.

Figures in "TCGI_singleyear_check_ERA5.pdf" show the differences between results with and without the refill-land process, demonstrating that our adjustments only affect coastal regions.

## References

> Camargo, Suzana J., et al. "Testing the performance of tropical cyclone genesis indices in future climates using the HiRAM model." Journal of Climate 27.24 (2014): 9171-9196.

> Gilford, D. M.: pyPI (v1.3): Tropical Cyclone Potential Intensity Calculations in Python, Geosci. Model Dev., 14, 2351–2369, https://doi.org/10.5194/gmd-14-2351-2021, 2021.

> Gilford, D. M. 2020: pyPI: Potential Intensity Calculations in Python, pyPI v1.3. Zenodo. http://doi.org/10.5281/zenodo.3985975

