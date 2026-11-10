#!/usr/bin/env python3
"""Compare two photometry catalogs by sky position and flux."""

from pathlib import Path

import matplotlib.pyplot as pl
pl.ion()
import numpy as np
from astropy.table import Table
import os

from rphot.io import get_psf_file
from rphot.apcorr import get_apcorr_from_psf
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter as gf



phot_dir = Path("/Users/ri3e/cv/galaxies/phot")
cat1_path = phot_dir / "ngc3351_convolved_pah_f360m_phot_cat_r2.86.ecsv"
cat2_path = phot_dir / "ngc3351_pahfind/ngc3351_convolved_pah_f360m_phot_cat_r2.86.ecsv"
out_plot  = phot_dir / "ngc3351_postTAB_flux_compare.png"
flux_col = "aperture_flux_mJy"
flux_col2 = "aperture_flux_mJy"

band = "F335M"
#band = "convolved_pah_f360m"

phot_dir = Path("/Users/ri3e/cv/galaxies/phot/indep_bands/")
out_plot  = phot_dir / f"ngc3351_app_psf_{band}_compare.png"
cat1_path = phot_dir / f"ngc3351_{band}_phot_cat.ecsv"

phot_dir = Path("/Users/ri3e/cv/galaxies/phot/from_hst/")
out_plot  = phot_dir / f"ngc3351_app_psf_{band}_from_hst_compare.png"
cat1_path = phot_dir / f"ngc3351_{band}_phot_cat_from_hst.ecsv"


cat2_path = cat1_path # sometimes we want different catalogs
flux_col = "aperture_flux"
flux_col2 = "psffit_ampwid_flux" 



cat1 = Table.read(cat1_path)
cat2 = Table.read(cat2_path)

if len(cat1) != len(cat2):
    raise ValueError(f"Catalog lengths differ: {len(cat1)} vs {len(cat2)}")

u=np.argsort(cat1['id'])
cat1 = cat1[u]
u=np.argsort(cat2['id'])
cat2 = cat2[u]

ra_diff_arcsec = np.abs(np.asarray(cat1["ra"], dtype=float) - np.asarray(cat2["ra"], dtype=float)) * 3600.0
dec_diff_arcsec = np.abs(np.asarray(cat1["dec"], dtype=float) - np.asarray(cat2["dec"], dtype=float)) * 3600.0

print(f"Max |delta RA|  (arcsec): {np.nanmax(ra_diff_arcsec):.6e}")
print(f"Max |delta Dec| (arcsec): {np.nanmax(dec_diff_arcsec):.6e}")
assert np.nanmax(ra_diff_arcsec) < 1.0, "RA difference too large"
assert np.nanmax(dec_diff_arcsec) < 1.0, "Dec difference too large"

f1 = np.asarray(cat1[flux_col], dtype=float)
f2 = np.asarray(cat2[flux_col2], dtype=float)

good=np.where((f1 > 1e-5) & (f2 > 1e-5))[0]
pl.clf()
pl.subplot(211)
pl.plot(f1[good], (f2/f1)[good], "k.", ms=2)
#ax.axhline(1.0, color='r', linestyle='--', lw=1)

z=np.where((cat2['psffit_ampwid_wout']<1e-5) & (f1 > 1e-5) & (f2 > 1e-5))[0]
pl.plot(f1[z],(f2/f1)[z], "r.", ms=2)

pl.xscale("log")
pl.yscale("log")
pl.xlim(1e-5, .1)
pl.ylim(.3,10)
pl.xlabel("Flux from aperture catalog (mJy)")
pl.ylabel("PSF-fit/aperture")
pl.title(f"{band}")


xparm=cat2['psffit_ampwid_wout']/4 # quarter-pix to pix

pl.subplot(212)
pl.plot(xparm, f2/f1, "k.", ms=2)
#ax.axhline(1.0, color='r', linestyle='--', lw=1)
    
psf_dir = '/Users/ri3e/cv/jwst_clusters/instrument/psf/'
psffile = get_psf_file(band,psf_dir=psf_dir)

psfhdu = fits.open(psffile)[0]
psfdat = psfhdu.data

widths=[0,2,6,10]     
widths=np.arange(10) # change manually to match whatever was done in residuals.py
nbroad=len(widths)

s=psfdat.shape
# this "p" array will store broadened psfs, at quarter-pix resolution
p=np.zeros([nbroad,s[0],s[1]]) 

# ROTATE psf to the correct position angle
#if 'pa_aper' in inhdu.header:
#    psfdat=rotate(psfdat,-inhdu.header['pa_aper'],reshape=False)
#else:
#    print("WARNING: can't rotate PSF because image position angle not found in header.")

apcorrs=np.zeros(nbroad, dtype=float)
for k in range(nbroad):
    p[k]=gf(psfdat,widths[k])
    # p[k]=gf(psfdat,widths[k]).reshape(s[0]//4,4,s[1]//4,4).sum(3).sum(1)
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    p[k]/=p[k].sum()
    psfhdu.data=p[k]
    psfkfile = psffile.split(".fits")[0] + f"_gauss_wid{widths[k]}.fits"
    if not os.path.exists(psfkfile):
        psfhdu.writeto(psfkfile)
    # integrate broadened PSFS inside of the aperture used - 1/that gives the factor by which the psffit phot should be > the apphot 
    apcorrs[k] = get_apcorr_from_psf(band, 
                                     r_ap=cat1.meta['ap_radius_pix'], 
                                     r_sky_in=cat1.meta['sky_in_radius_pix'], 
                                     r_sky_out=cat1.meta['sky_out_radius_pix'], 
                                     psf_file=psfkfile)


pl.plot(np.array(widths[1:])/4, (np.array([apcorrs[k] for k in range(1, nbroad)])))

pl.xlabel("PSF-fit width (pixel)")
pl.ylabel("PSF-fit/aperture")



pl.tight_layout()
pl.savefig(out_plot)


print(f"Saved plot: {out_plot}")
