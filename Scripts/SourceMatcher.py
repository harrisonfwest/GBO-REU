#!/users/lmorgan/bin/PythonVenvs/venv312/bin/python3.12
import requests
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

def ks_cost(factor):
    scaled = factor * np.array(SizeClF.ravel())
    stat, _ = ks_2samp(scaled, SizeRAMPS)
    return stat

# Define the cost function: MSE between scaled ClF sizes and RAMPS sizes
def mse_cost(factor):
    scaled = factor * np.array(SizeClF.ravel())
    return np.mean((scaled - SizeRAMPS)**2)

def ks_costdv(factor):
    scaled = factor * np.array(SigClF.ravel())
    stat, _ = ks_2samp(scaled, SigRAMPS)
    return stat

# Define the cost function: MSE between scaled ClF sizes and RAMPS sizes
def mse_costdv(factor):
    scaled = factor * np.array(SigClF.ravel())
    return np.mean((scaled - SigRAMPS)**2)

# File paths
# Tile = 'L28_5'
tiles = ['L10', 'L10_5', 'L11', 'L11_5', 'L12', 'L12_5', 'L13', 'L13_5', 'L14', 'L14_5', 'L15', 'L15_5', 'L16', 
         'L16_5', 'L17', 'L17_5', 'L18', 'L18_5', 'L19', 'L19_5', 'L20', 'L20_5', 'L21', 'L21_5', 'L22', 'L22_5', 
         'L23', 'L23_5', 'L24', 'L24_5', 'L25', 'L25_5', 'L26', 'L26_5', 'L27', 'L27_5', 'L28', 'L28_5', 'L29', 
         'L29_5', 'L30', 'L30_5', 'L31', 'L31_5', 'L32', 'L32_5', 'L33', 'L33_5', 'L34', 'L34_5', 'L35', 'L35_5', 
         'L36', 'L36_5', 'L37', 'L37_5', 'L38', 'L38_5', 'L39', 'L39_5', 'L40', 'L40_5', 'L41']
#path = '/Users/lmorgan/Documents/Projects/RAMPS/'
path = '/home/scratch/lmorgan/Projects/RAMPS/'
RAMPSClumpCat = path + 'Data/RAMPS_Clump_Catalogue.txt'
ClFClumpCat = path + 'Data/RAMPS_ClF_Catalogue.txt'

all_size_factors_MSE = []
all_size_factors_leastSq = []
all_SigV_factors_MSE = []
all_SigV_factors_leastSq = []

_plot = False # Whether or not to show all plots for each tile (interrupts script)

for Tile in tiles:
    dfRAMPS=pd.read_csv(RAMPSClumpCat,sep=r'\s+',comment='#',header=None,skiprows=1)
    dfRAMPS.columns=['ClumpName','Tile','Mask','l_cen','b_cen','Clump_RMS','Clump_Peak_T','l_peak','b_peak','Source_Peak_RMS','Source_Peak_T','theta_maj','theta_min','sigma','sigma_err','V_LSR']
    NameRAMPS=np.array(dfRAMPS['ClumpName'],dtype=str)
    TileRAMPS=np.array(dfRAMPS['Tile'],dtype=str)
    MaskRAMPS=np.array(dfRAMPS['Mask'],dtype=int)
    lRAMPS = np.array(dfRAMPS['l_cen'],dtype=float)
    bRAMPS = np.array(dfRAMPS['b_cen'],dtype=float)
    SzRAMPS = np.array(dfRAMPS['theta_maj'], dtype=float)
    SigVRAMPS = np.array(dfRAMPS['sigma'],dtype=float)
    PeakTRAMPS = np.array(dfRAMPS['Source_Peak_T'],dtype=float)
    PeakRMSRAMPS = np.array(dfRAMPS['Source_Peak_RMS'],dtype=float)
    galcoordsRAMPS = SkyCoord(l=lRAMPS*u.deg,b=bRAMPS*u.deg,frame='galactic')

    dfClF=pd.read_csv(ClFClumpCat,sep=r'\s+',comment='#',header=None,skiprows=1)
    dfClF.columns=['ClumpName','Tile','Mask','l_cen','b_cen','Clump_RMS','Clump_Peak_T','l_peak','b_peak','Source_Peak_RMS','Source_Peak_T','theta_maj','theta_min','sigma','sigma_err','V_LSR']
    NameClF=np.array(dfClF['ClumpName'],dtype=str)
    TileClF=np.array(dfClF['Tile'],dtype=str)
    MaskClF=np.array(dfClF['Mask'],dtype=int)
    lClF = np.array(dfClF['l_cen'],dtype=float)
    bClF = np.array(dfClF['b_cen'],dtype=float)
    SzClF = np.array(dfClF['theta_maj'], dtype=float)
    SigVClF = np.array(dfClF['sigma'],dtype=float)
    PeakTClF = np.array(dfClF['Source_Peak_T'],dtype=float)
    PeakRMSClF = np.array(dfClF['Source_Peak_RMS'],dtype=float)
    galcoordsClF = SkyCoord(l=lClF*u.deg,b=bClF*u.deg,frame='galactic')

    # Filter to tile
    RAMPSmsk = TileRAMPS == Tile
    ClFmsk = TileClF == Tile
    NameRAMPSClip = NameRAMPS[RAMPSmsk]
    TileRAMPSClip = TileRAMPS[RAMPSmsk]
    MaskRAMPSClip = MaskRAMPS[RAMPSmsk]
    SzRAMPSClip = SzRAMPS[RAMPSmsk]
    SigVRAMPSClip = SigVRAMPS[RAMPSmsk]
    PeakTRAMPSClip = PeakTRAMPS[RAMPSmsk]
    PeakRMSRAMPSClip = PeakRMSRAMPS[RAMPSmsk]
    NameClFClip = NameClF[ClFmsk]
    TileClFClip = TileClF[ClFmsk]
    MaskClFClip = MaskClF[ClFmsk]
    SzClFClip = SzClF[ClFmsk]
    SigVClFClip = SigVClF[ClFmsk]
    PeakTClFClip = PeakTClF[ClFmsk]
    PeakRMSClFClip = PeakRMSClF[ClFmsk]
    TileRMS = np.median(PeakRMSRAMPSClip)

    RampsWClF = []
    RampsNoClF = []
    RAMPSUnmatchedTs = []
    RAMPSMatchedTs = []
    RAMPSUnmatchedSigVs = []
    RAMPSMatchedSigVs = []
    RAMPSUnmatchedSizes = []
    RAMPSMatchedSizes = []
    for Rijk in range(len(NameRAMPSClip)):
        if NameRAMPSClip[Rijk] not in NameClFClip:
            RampsNoClF.append(NameRAMPSClip[Rijk])
            RAMPSUnmatchedTs.append(PeakTRAMPSClip[Rijk])
            RAMPSUnmatchedSigVs.append(SigVRAMPSClip[Rijk])
            RAMPSUnmatchedSizes.append(SzRAMPSClip[Rijk])
        else:
            RampsWClF.append(NameRAMPSClip[Rijk])
            RAMPSMatchedTs.append(PeakTRAMPSClip[Rijk])
            RAMPSMatchedSigVs.append(SigVRAMPSClip[Rijk])
            RAMPSMatchedSizes.append(SzRAMPSClip[Rijk])
    RAMPSUnmatchedTs = np.array(RAMPSUnmatchedTs)
    RAMPSMatchedTs = np.array(RAMPSMatchedTs)
    RAMPSUnmatchedSigVs = np.array(RAMPSUnmatchedSigVs)
    RAMPSMatchedSigVs = np.array(RAMPSMatchedSigVs)

    ClFNoRAMPS = []
    for ClFSource in NameClFClip:
        if ClFSource not in NameRAMPSClip:
            ClFNoRAMPS.append(ClFSource)

    print('Tile - ',Tile)
    print('No. of RAMPS clumps - ',len(NameRAMPSClip))
    print('No. of ClumpFind clumps - ',len(NameClFClip))
    print('RAMPS with no ClF - ', len(RampsNoClF))
    print('ClF with no RAMPS - ', len(ClFNoRAMPS))

    # Plot histogram of Peak Ts
    # Compute histograms without plotting
    counts_all, bins = np.histogram(PeakTRAMPSClip, bins=20)
    counts_match, _ = np.histogram(RAMPSMatchedTs, bins=bins)
    counts_nonmatch, _ = np.histogram(RAMPSUnmatchedTs, bins=bins)

    # Determine the scaling factor
    scale_factor_match = counts_all.max() / counts_match.max()
    scale_factor_nonmatch = counts_all.max() / counts_nonmatch.max()
    print(f"{len(RAMPSMatchedTs)}")
    print(f"{len(RAMPSMatchedTs[np.where(RAMPSMatchedTs > 5*TileRMS)])}")
    print(f"{len(PeakTRAMPSClip[np.where(PeakTRAMPSClip > 5*TileRMS)])}")
    print(f"{len(RAMPSMatchedTs)*100/len(PeakTRAMPSClip):4.1f}% of RAMPS sources matched by ClumpFind")
    print(f"{len(RAMPSMatchedTs[np.where(RAMPSMatchedTs > 5*TileRMS)])*100/len(PeakTRAMPSClip[np.where(PeakTRAMPSClip > 5*TileRMS)]):4.1f}% of RAMPS sources matched by ClumpFind above 5 sigma")
    #print(5*TileRMS)
    # Plot the histograms with matched peak heights
    plt.figure(figsize=(8, 5))
    plt.hist(PeakTRAMPSClip, bins=bins, color='black', edgecolor='black',label='All RAMPS')
    plt.hist(RAMPSMatchedTs, bins=bins, weights=np.ones_like(RAMPSMatchedTs) * scale_factor_match, color='green', edgecolor='black', alpha=0.3,label='Matched')
    plt.hist(RAMPSUnmatchedTs, bins=bins, weights=np.ones_like(RAMPSUnmatchedTs) * scale_factor_nonmatch, color='red', edgecolor='black', alpha=0.3,label='Unmatched')
    plt.axvline(x=5*TileRMS, color='red', linestyle='--', linewidth=2, label='Threshold')
    plt.legend()
    plt.title("Histogram of Peak Ts (K)")
    plt.xlabel("Peak Ts")
    plt.ylabel("(Normalised) Number of Matches")
    plt.grid(True)
    plt.tight_layout()


    # Plot histogram of Sig Vs
    # Compute histograms without plotting
    counts_all, bins = np.histogram(SigVRAMPSClip, bins=20)
    counts_match, _ = np.histogram(RAMPSMatchedSigVs, bins=bins)
    counts_nonmatch, _ = np.histogram(RAMPSMatchedSigVs, bins=bins)

    # Determine the scaling factor
    scale_factor_match = counts_all.max() / counts_match.max()
    scale_factor_nonmatch = counts_all.max() / counts_nonmatch.max()
    # Plot the histograms with matched peak heights
    plt.figure(figsize=(8, 5))
    plt.hist(SigVRAMPSClip, bins=bins, color='black', edgecolor='black',label='All RAMPS')
    plt.hist(RAMPSMatchedSigVs, bins=bins, weights=np.ones_like(RAMPSMatchedSigVs) * scale_factor_match, color='green', edgecolor='black', alpha=0.3,label='Matched')
    plt.hist(RAMPSMatchedSigVs, bins=bins, weights=np.ones_like(RAMPSMatchedSigVs) * scale_factor_nonmatch, color='red', edgecolor='black', alpha=0.3,label='Unmatched')
    plt.legend()
    plt.title("Histogram of Sig V's (km/s)")
    plt.xlabel("Sig V")
    plt.ylabel("(Normalised) Number of Matches")
    plt.grid(True)
    plt.tight_layout()

    # Load ClumpFind catalog
    ClFindFile = os.path.join(path, 'ClumpFind', 'LogFiles', f"{Tile}.log")
    dfClF = pd.read_csv(ClFindFile, skiprows=[0,1,2,4], skipfooter=25, engine='python', sep=r'\s+')

    # Extract CUPID ClumpFind values
    idxClF = np.array(dfClF['Index'])
    xClF = np.array(dfClF['Cen1'])
    yClF = np.array(dfClF['Cen2'])
    vClF = np.array(dfClF['Cen3'])
    FWHMxClF = np.array(dfClF['Size1'])
    FWHMyClF = np.array(dfClF['Size2'])
    FWHMvClF = np.array(dfClF['Size3'])
    PeakClF = np.array(dfClF['Peak'])
    VolClF = np.array(dfClF['Volume'])

    ClFN = []
    ClFName = []
    PeakTClF = []
    SizeClF = []
    SigClF = []
    SizeRAMPS = []
    SigRAMPS = []

    for RCijk in range(len(RampsWClF)):
        flg = np.where(NameClFClip == RampsWClF[RCijk])
        msk = MaskClFClip[flg]
        mskflg = np.where(idxClF == msk)
        ClFName.append(NameClFClip[flg])
        ClFN.append(idxClF[mskflg])
        xClFInd = xClF[mskflg]
        yClFInd = yClF[mskflg]
        vClFInd = vClF[mskflg]
        FWHMxClFInd = FWHMxClF[mskflg]
        FWHMyClFInd = FWHMyClF[mskflg]
        FWHMvClFInd = FWHMvClF[mskflg]
        VolClFInd = VolClF[mskflg]
        R1 = min(FWHMxClFInd,FWHMyClFInd)/max(FWHMxClFInd,FWHMyClFInd)
        R2 = FWHMvClFInd/max(FWHMxClFInd,FWHMyClFInd)
        SizeClF.append(((3*VolClFInd/(4*np.pi*R1*R2))**(1/3))*2*6/60)
        SigClF.append(FWHMvClFInd*0.2/(2*np.sqrt(2*np.log(2))))
        PeakTClF.append(PeakClF[mskflg])
        SizeRAMPS.append(RAMPSMatchedSizes[RCijk])
        SigRAMPS.append(RAMPSMatchedSigVs[RCijk])

    # for kji in range(len(SizeClF)):
    #     print(ClFName[kji],ClFN[kji],SizeClF[kji],SigClF[kji],PeakTClF[kji],SizeRAMPS[kji],SigRAMPS[kji])
    sf = 1.2472
    sf = 1.0
    SizeClF = np.array(SizeClF)*sf
    SizeRAMPS = np.array(SizeRAMPS)
    SigClF = np.array(SigClF)
    SigRAMPS = np.array(SigRAMPS)
    szratio = SizeClF/SizeRAMPS
    szratio = szratio[0]
    if len(szratio) > 0:
        print("\nSummary Statistics for SizeClF / SzRAMPSClip:")
        print(f"  Min   : {np.min(szratio):.3f}")
        print(f"  Max   : {np.max(szratio):.3f}")
        print(f"  Mean  : {np.mean(szratio):.3f}")
        print(f"  Median: {np.median(szratio):.3f}")

        # Plot histogram of ratios
        plt.figure(figsize=(8, 5))
        plt.hist(szratio, bins=20, color='blue', edgecolor='black')
        plt.title("Histogram of ClumpFind / RAMPS Size Ratios")
        plt.xlabel("Size ClF / Size RAMPS")
        plt.ylabel("(Normalised) Number of Matches")
        plt.grid(True)
        plt.tight_layout()

        # Scatter: SizeClF vs SizeRAMPS (log-log)
        plt.figure(figsize=(8, 5))
        plt.scatter(SizeRAMPS, SizeClF.ravel(), color='cornflowerblue')
        plt.title("Size ClF vs Size RAMPS")
        plt.xlabel("Size RAMPS")
        plt.ylabel("Size ClF")
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

        # Histogram: ClumpFind sizes
        counts_clf, bins = np.histogram(SizeClF.ravel(), bins=20)
        counts_ramps, _ = np.histogram(SizeRAMPS, bins=bins)

    # Determine the scaling factor
        scale_factor = counts_clf.max() / counts_ramps.max()

        plt.figure(figsize=(8, 5))
        plt.hist(SizeClF.ravel(), bins=20, color='steelblue', edgecolor='black', alpha=0.7, label='ClF')
        plt.hist(SizeRAMPS, bins=bins, weights=np.ones_like(SizeRAMPS) * scale_factor, color='red', edgecolor='black', alpha=0.7,label='RAMPS')
        plt.title("Histogram of RAMPS and ClF Sizes")
        plt.xlabel("Size")
        plt.ylabel("(Normalised) Number of Matches")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # KS test
        ks_stat, p_val = ks_2samp(SizeClF.ravel(), SizeRAMPS)
        print("\nKolmogorov-Smirnov Test for Sizes:")
        print(f"  KS Statistic: {ks_stat:.3f}")
        print(f"  p-value     : {p_val:.3e}")
        if p_val < 0.05:
            print("  -> Distributions differ significantly at the 5% level.")
        else:
            print("  -> No statistically significant difference at the 5% level.")

        result_ks = minimize_scalar(ks_cost, bounds=(0.1, 10), method='bounded')
        opt_ks_factor = result_ks.x

        print(f"Optimal scale factor (KS test): {opt_ks_factor:.4f}")

        # Find the optimal scaling factor
        result = minimize_scalar(mse_cost, bounds=(0.1, 10), method='bounded')
        opt_factor = result.x

        print(f"\nOptimal scale factor (MSE): {opt_factor:.4f}")
        model = LinearRegression(fit_intercept=False)
        model.fit(np.array(SizeClF).reshape(-1, 1), SizeRAMPS)
        scale_lstsq = model.coef_[0]

        print(f"Optimal scale factor (least squares): {scale_lstsq:.4f}")

        all_size_factors_MSE.append(opt_factor)
        all_size_factors_leastSq.append(scale_lstsq)

        sorted_clf, cdf_clf = compute_cdf(SizeClF.ravel())
        sorted_ramps, cdf_ramps = compute_cdf(SizeRAMPS)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_clf, cdf_clf, label='ClumpFind', color='steelblue')
        plt.plot(sorted_ramps, cdf_ramps, label='RAMPS', color='salmon')
        plt.title("Cumulative Distribution Functions")
        plt.xlabel("Angular Size")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Scatter: SigVClF vs SigVRAMPS (log-log)
        plt.figure(figsize=(8, 5))
        plt.scatter(SigRAMPS, SigClF.ravel(), color='cornflowerblue')
        plt.title("SigV ClF vs SigV RAMPS")
        plt.xlabel("SigV RAMPS")
        plt.ylabel("SigV ClF")
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

        # Histogram: ClumpFind sig Vs
        counts_clf, bins = np.histogram(SigClF, bins=20)
        counts_ramps, _ = np.histogram(SigRAMPS, bins=bins)

    # Determine the scaling factor
        scale_factor = counts_clf.max() / counts_ramps.max()

        plt.figure(figsize=(8, 5))
        plt.hist(SigClF.ravel(), bins=20, color='steelblue', edgecolor='black', alpha=0.7, label='ClF')
        plt.hist(SigRAMPS, bins=bins, weights=np.ones_like(SigRAMPS) * scale_factor, color='red', edgecolor='black', alpha=0.7,label='RAMPS')
        plt.title("Histogram of RAMPS/ClF Sig Vs")
        plt.xlabel("Size")
        plt.ylabel("(Normalised) Number of Matches")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # KS test
        ks_stat, p_val = ks_2samp(SigClF.ravel(), SigRAMPS)
        print(ks_stat)
        print("\nKolmogorov-Smirnov Test for SigVs:")
        print(f"  KS Statistic: {ks_stat:.3f}")
        print(f"  p-value     : {p_val:.3e}")
        if p_val < 0.05:
            print("  -> Distributions differ significantly at the 5% level.")
        else:
            print("  -> No statistically significant difference at the 5% level.")

        result_ks = minimize_scalar(ks_costdv, bounds=(0.1, 10), method='bounded')
        opt_ks_factor = result_ks.x

        print(f"Optimal scale factor (KS test): {opt_ks_factor:.4f}")

        # Find the optimal scaling factor
        result = minimize_scalar(mse_costdv, bounds=(0.1, 10), method='bounded')
        opt_factor = result.x

        print(f"\nOptimal scale factor (MSE): {opt_factor:.4f}")

        model = LinearRegression(fit_intercept=False)
        model.fit(np.array(SigClF).reshape(-1, 1), SigRAMPS)
        scale_lstsq = model.coef_[0]

        print(f"Optimal scale factor (least squares): {scale_lstsq:.4f}")

        all_SigV_factors_MSE.append(opt_factor)
        all_SigV_factors_leastSq.append(scale_lstsq)

        sorted_clf, cdf_clf = compute_cdf(SigClF.ravel())
        sorted_ramps, cdf_ramps = compute_cdf(SigRAMPS)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_clf, cdf_clf, label='ClumpFind', color='steelblue')
        plt.plot(sorted_ramps, cdf_ramps, label='RAMPS', color='salmon')
        plt.title("Cumulative Distribution Functions")
        plt.xlabel("Sigma V")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if _plot:
            plt.show()
    else:
        print("No size ratios to summarize or plot.")

# print(f'Size scale factors (MSE): {all_size_factors_MSE}')
# print(f'Size scale factors (least squares): {all_size_factors_leastSq}')
print(f'Mean size factor (MSE): {np.mean(all_size_factors_MSE)}')
print(f'Mean size factor (Least Squares): {np.mean(all_size_factors_leastSq)}')
# print(f'SigV scale factors (MSE): {all_SigV_factors_MSE}')
# print(f'SigV scale factors (least squares): {all_SigV_factors_leastSq}')
print(f'Mean SigV factor (MSE): {np.mean(all_SigV_factors_MSE)}')
print(f'Mean SigV factor (Least Squares): {np.mean(all_SigV_factors_leastSq)}')