1. Perform fitting on Butterfield data (NH3 1,1 and 2,2); construct single gaussian model. Source finding can then be applied to that model in order to create a mask of where sources are located in the structure observed by Butterfield+
2. It's not been decided yet exactly how the source-finding will be done but it would be useful to do the exact same thing that was done for the RAMPS data, so that we can do a side-by-side analysis. There's no need for you to jump on to this right now but it would be a good idea for you to read through the thesis I gave you and look at the way that Taylor did these things for the RAMPS data. If you think you can figure it out, it would be good to try and perform the same source-finding/identification procedures on the Midpoint data. You can find the ammonia (1,1) cube at /home/scratch/lmorgan/Projects/GalacticBar/Data/Pilot_All/Pilot_All/Pilot_NH3_11_bl2.fits
3. Learn about ammonia spectra: (J,K) transition meaning (spin, projection of spin onto axis of molecule); Butterfield et al, 2025 has spectra for various transitions

FWHM = 2 sqrt(2 * ln(2)) * sigma; FWHM ~ 2.35 * sigma

Butterfield et al, 2025 Sources:
1. Oka, T., Hasegawa, T., Sato, F., Tsuboi, M., Miyazaki, A.,
& Sugimoto, M. 2001b, ApJ, 562, 348: https://ui.adsabs.harvard.edu/abs/2001ApJ...562..348O/abstract
  * Presents *velocity dispersion*
2. Kauffmann, J., Pillai, T., Zhang, Q., Menten, K. M.,
Goldsmith, P. F., Lu, X., & Guzm ́an, A. E. 2017, A&A,
603, A89, 1610.03499: https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..89K/abstract
  * Presents *velocity dispersion*
3. Heyer, M., Krawczyk, C., Duval, J., & Jackson, J. M. 2009,
ApJ, 699, 1092, 0809.1397: https://ui.adsabs.harvard.edu/abs/2009ApJ...699.1092H/abstract
  * Presents *velocity dispersion* as reported in Solomon et al. 1987 (SRBY)
4. Urquhart, J. S. et al. 2015, MNRAS, 452, 4029, 1507.02187: https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.4029U/abstract
  * Presents *line widths (FWHM)*
5. Veena, V. S. et al. 2024, A&A, 689, A121, 2407.14338: https://ui.adsabs.harvard.edu/abs/2024A%26A...689A.121V/abstract
  * Presents *line widths (FWHM)*


Tuesday: Cross-reference Urquhart with RAMPS (matching by location via. LM script in slack within ~30 arcsec); compare two catalogs' values for R, sigma (i.e. co-plot), and eventually glue all back together.