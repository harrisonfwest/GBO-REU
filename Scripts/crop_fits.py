from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

file = '/home/scratch/hfwest/Pilot/Data/Pilot_NH3_11_bl.fits'
with fits.open(file) as hdul:
    data = hdul[0].data[0]
    header = hdul[0].header

left = 15
right = 22
top = 15
bottom = 15

cropped_image = data[:, top:-bottom, left:-right]
hdu = fits.PrimaryHDU(cropped_image, header= header)
hdu.writeto('/home/scratch/hfwest/Pilot/Data/Pilot_NH3_11_cropped.fits', overwrite= True)