import numpy as np
import astropy.units as u
import warnings
import os
from astropy.wcs import WCS
from astropy.io import fits
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube

nu11 = 23.6944955e9

Cube = SpectralCube.read('/home/scratch/hfwest/Pilot/Data/Pilot_NH3_11_cropped.fits')
Cube = Cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=nu11 * u.Hz)

vl = 150 * u.km / u.s
vh = 240 * u.km / u.s

Cube_slab = Cube.spectral_slab(vl, vh)
cube_data = Cube_slab.unmasked_data[:].value
vel_axis = Cube_slab.spectral_axis

nz, ny, nx = Cube.shape

peak = 0

# indices = [(j, i) for j in range(ny) for i in range(nx)]
for j in range(ny):
    for i in range(nx):
        if np.nanmax(cube_data[:, j, i]) >= peak:
            peak = np.nanmax(cube_data[:, j, i])

print(peak)