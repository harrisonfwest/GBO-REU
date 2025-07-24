#!/users/lmorgan/bin/PythonVenvs/optpy/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import warnings
import os
from astropy.wcs import WCS
from astropy.io import fits
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Constants
etamb = 0.89
etaf = 1.0
nu11 = 23.6944955e9
h = 6.62606896e-34
kB = 1.3806504e-23
Tbg = 2.73
c = 2.99792458e8
To = 41.5

# Derived constants
mu = 1.476 * 3.336e-30
mu11 = mu ** 2 / 2.
epsilon = 8.854187817e-12
Einstein_A = (16. * np.pi ** 3 / (3. * epsilon * h * c ** 3)) * nu11 ** 3 * abs(mu11)

# Velocity bounds
vmargin = 50.
threshold = 3
min_peak_separation_chan = 3
hf_lv = 0.5
vl = 140 * u.km / u.s
vh = 260 * u.km / u.s
rmsvl = 280 * u.km / u.s
rmsvh = 320 * u.km / u.s

# Model functions
def gaussian(x, amp, vel, sigma):
    return amp * np.exp(-(x - vel)**2 / (2 * sigma**2))

def quad_func(x, amp0, vel0, sigma0, tau0):
    amp = [0.226, 0.273, 1.0, 0.277, 0.219]
    vel = [-19.503, -7.594, 0.0, 7.599, 19.493]
    emission_profile = np.zeros_like(x)
    for ampi, veli in zip(amp, vel):
        Tq = (1 - np.exp(-ampi * tau0)) / (1 - np.exp(-tau0))
        emission_profile += gaussian(x, Tq, vel0 + veli, sigma0)
    return amp0 * emission_profile

def double_quad_func(x, amp0, vel0, sigma0, tau0, amp1, vel1, sigma1, tau1):
    return quad_func(x, amp0, vel0, sigma0, tau0) + quad_func(x, amp1, vel1, sigma1, tau1)

def quadrupole_to_peakmatched_gaussian(popt, vel_axis):
    A0, v0, sigma0, tau0 = popt
    quad = quad_func(vel_axis, A0, v0, sigma0, tau0)
    peak = np.max(quad)
    gauss = peak * np.exp(-(vel_axis - v0)**2 / (2 * sigma0**2))
    return gauss

# Load data
path = '/home/scratch/hfwest/Pilot/'
outdir = path + 'Results/'
suffixout = 'Pool'
Cube = SpectralCube.read(path + 'Data/Pilot_NH3_11_cropped.fits')
prefixout = 'Pilot_NH3_11'

Cube = Cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=nu11 * u.Hz)
Cube_slab = Cube.spectral_slab(vl, vh)
RMS_slab = Cube.spectral_slab(rmsvl, rmsvh)
cube_data = Cube_slab.unmasked_data[:].value
rms_data = RMS_slab.unmasked_data[:].value
vel_axis = Cube_slab.spectral_axis

nz, ny, nx = Cube.shape

header_3d = Cube.wcs.to_header()
header_2d = Cube[Cube.shape[0] // 2].wcs.to_header()
for key in ['BUNIT', 'OBJECT', 'TELESCOP', 'DATE-OBS']:
    if key in Cube.header:
        header_2d[key] = Cube.header[key]

full_vel_axis = Cube.spectral_axis.value

# Fit function per pixel
def fit_pixel(j, i):
    spec = cube_data[:, j, i]
    rms = np.std(rms_data[:, j, i])
    if np.max(spec) < 2 * rms:
        return (0, j, i, None)

    peaks, props = find_peaks(spec, height=threshold * rms, distance=min_peak_separation_chan)
    if len(peaks) == 0:
        return (0, j, i, None)

    sorted_idx = np.argsort(props['peak_heights'])[::-1]
    pmax = peaks[sorted_idx[0]]
    amp0 = spec[pmax]
    vel0 = vel_axis[pmax].value

    mask = (vel_axis.value > vel0 - vmargin) & (vel_axis.value < vel0 + vmargin)
    x = vel_axis[mask].value
    y = spec[mask]

    try:
        p1, c1 = curve_fit(quad_func, x, y, p0=[amp0, vel0, 2.0, 1.0], bounds=([threshold*rms, vel0-10, 0.01, 0.01], [amp0*2, vel0+10, 50, 50]))
        residuals1 = y - quad_func(x, *p1)
        chi1 = np.sum((residuals1 / rms)**2)
        dof1 = len(y) - 4
        bic1 = chi1 + 4 * np.log(len(y))

        if len(peaks) > 1:
            pmax2 = peaks[sorted_idx[1]]
            amp1 = spec[pmax2]
            vel1 = vel_axis[pmax2].value
            if amp1 > amp0*0.5:
                p2, c2 = curve_fit(double_quad_func, x, y,
                                   p0=[amp0, vel0, 2.0, 1.0, amp1, vel1, 2.0, 1.0],
                                   bounds=([threshold*rms, vel0-10, 0.01, 0.01, threshold*rms, vel1-10, 0.01, 0.01],
                                           [amp0*2, vel0+10, 50, 50, amp1*2, vel1+10, 50, 50]))
                residuals2 = y - double_quad_func(x, *p2)
                chi2 = np.sum((residuals2 / rms)**2)
                bic2 = chi2 + 8 * np.log(len(y))

                if bic2 < bic1:
                    yfit = double_quad_func(full_vel_axis, *p2)
                    return (2, j, i, p2, yfit)

        yfit = quad_func(full_vel_axis, *p1)
        return (1, j, i, p1, yfit)

    except RuntimeError:
        return (0, j, i, None)

# Multiprocessing dispatcher
def run_parallel():
#     xpix = 87
#     ypix = 79   #two-component
#     # xpix = 99
#     # ypix = 85   #single-component
    indices = [(j, i) for j in range(ny) for i in range(nx)]
#     indices = [(j, i) for j in [ypix] for i in [xpix]]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(fit_pixel, indices)

    FitCube = np.full((nz, ny, nx), np.nan)
    GaussCube = np.full((nz, ny, nx), np.nan)
    NCompMap = np.full((ny, nx), np.nan)
    TMax1cMap = np.full((ny, nx), np.nan)
    Vel1cMap = np.full((ny, nx), np.nan)
    Sigma1cMap = np.full((ny, nx), np.nan)
    Tau1cMap = np.full((ny, nx), np.nan)

    TMax2c1Map = np.full((ny, nx), np.nan)
    Vel2c1Map = np.full((ny, nx), np.nan)
    Sigma2c1Map = np.full((ny, nx), np.nan)
    Tau2c1Map = np.full((ny, nx), np.nan)
    TMax2c2Map = np.full((ny, nx), np.nan)
    Vel2c2Map = np.full((ny, nx), np.nan)
    Sigma2c2Map = np.full((ny, nx), np.nan)
    Tau2c2Map = np.full((ny, nx), np.nan)

    for res in results:
        if res[3] is None:
            continue
        if res[0] is 1:
            ncomp, j, i, popt, yfit = res
            NCompMap[j, i] = ncomp
            FitCube[:, j, i] = yfit
            g = quadrupole_to_peakmatched_gaussian(popt, full_vel_axis)
            GaussCube[:, j, i] = g
            TMax1cMap[j, i] =popt[0]
            Vel1cMap[j, i] = popt[1]
            Sigma1cMap[j, i] = popt[2]
            Tau1cMap[j, i] = popt[3]
        if res[0] is 2:
            ncomp, j, i, popt, yfit = res
            TMaxArr = [popt[0],popt[4]]
            VelArr = [popt[1],popt[5]]
            SigmaArr = [popt[2],popt[6]]
            TauArr = [popt[3],popt[7]]
            largeridx = TMaxArr.index(max(TMaxArr))
            smalleridx = TMaxArr.index(min(TMaxArr))
            g1 = quadrupole_to_peakmatched_gaussian(popt[largeridx*4:(largeridx*4)+4], full_vel_axis)
            g2 = quadrupole_to_peakmatched_gaussian(popt[smalleridx*4:(smalleridx*4)+4], full_vel_axis)
            NCompMap[j, i] = ncomp
            FitCube[:, j, i] = yfit
            GaussCube[:, j, i] = g1+g2
            TMax1cMap[j, i] = TMaxArr[largeridx]
            Vel1cMap[j, i] = VelArr[largeridx]
            Sigma1cMap[j, i] = SigmaArr[largeridx]
            Tau1cMap[j, i] = TauArr[largeridx]

            TMax2c1Map[j, i] = TMaxArr[largeridx]
            Vel2c1Map[j, i] = VelArr[largeridx]
            Sigma2c1Map[j, i] = SigmaArr[largeridx]
            Tau2c1Map[j, i] = TauArr[largeridx]
            TMax2c2Map[j, i] = TMaxArr[smalleridx]
            Vel2c2Map[j, i] = VelArr[smalleridx]
            Sigma2c2Map[j, i] = SigmaArr[smalleridx]
            Tau2c2Map[j, i] = TauArr[smalleridx]

    fits.PrimaryHDU(FitCube, header=header_3d).writeto(outdir + prefixout + '_Fit_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(GaussCube, header=header_3d).writeto(outdir + prefixout + '_GaussFit_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(NCompMap, header=header_2d).writeto(outdir + prefixout + '_NComp_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(TMax1cMap, header=header_2d).writeto(outdir + prefixout + '_TMax1c_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Vel1cMap, header=header_2d).writeto(outdir + prefixout + '_Vel1c_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Sigma1cMap, header=header_2d).writeto(outdir + prefixout + '_Sigma1c_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Tau1cMap, header=header_2d).writeto(outdir + prefixout + '_Tau1c_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(TMax2c1Map, header=header_2d).writeto(outdir + prefixout + '_TMax2c1_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Vel2c1Map, header=header_2d).writeto(outdir + prefixout + '_Vel2c1_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Sigma2c1Map, header=header_2d).writeto(outdir + prefixout + '_Sigma2c1_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Tau2c1Map, header=header_2d).writeto(outdir + prefixout + '_Tau2c1_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(TMax2c2Map, header=header_2d).writeto(outdir + prefixout + '_TMax2c2_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Vel2c2Map, header=header_2d).writeto(outdir + prefixout + '_Vel2c2_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Sigma2c2Map, header=header_2d).writeto(outdir + prefixout + '_Sigma2c2_' + suffixout + '.fits', overwrite=True)
    fits.PrimaryHDU(Tau2c2Map, header=header_2d).writeto(outdir + prefixout + '_Tau2c2_' + suffixout + '.fits', overwrite=True)

if __name__ == '__main__':
    run_parallel()
