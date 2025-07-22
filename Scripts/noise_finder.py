import numpy as np
import astropy.units as u
import warnings
import os
from astropy.wcs import WCS
from astropy.io import fits
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube

nu11 = 23.6944955e9

Cube = SpectralCube.read('/home/scratch/hfwest/Pilot/Data/Pilot_NH3_11_bl.fits')
Cube = Cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=nu11 * u.Hz)

rmsvl = 250 * u.km / u.s
rmsvh = 320 * u.km / u.s

RMS_slab = Cube.spectral_slab(rmsvl, rmsvh)
rms_data = RMS_slab.unmasked_data[:].value

nz, ny, nx = Cube.shape

arr = np.empty(shape= ny * nx)
count = 0

# indices = [(j, i) for j in range(ny) for i in range(nx)]
for j in range(ny):
    for i in range(nx):
        rms = np.std(rms_data[:, j, i])
        arr[count] = rms
        count += 1

mask = ~np.isnan(arr)

# Use the boolean mask to select only the non-NaN values from the array
clean_data = arr[mask]

print(np.median(clean_data)) # 0.012656566234247657