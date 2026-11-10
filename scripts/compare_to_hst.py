#!/usr/bin/env python3
"""Compare two photometry catalogs by sky position and flux."""

from pathlib import Path

import matplotlib.pyplot as pl
pl.ion()
import numpy as np
from astropy.table import Table


phot_dir = Path("/Users/ri3e/cv/galaxies/phot")
cat1_path = phot_dir / "ngc3351_combined.ecsv"
cat0_path = phot_dir / "Phot_v5p4_ngc3351_IR4_class12human.csv"

cat1 = Table.read(cat1_path)
cat0 = Table.read(cat0_path)

if len(cat1) != len(cat0):
    raise ValueError(f"Catalog lengths differ: {len(cat1)} vs {len(cat0)}")

u=np.argsort(cat1['id'])
cat1 = cat1[u]
u=np.argsort(cat0['ID_phangs'])
cat0 = cat0[u]

# naming issue temp fix
cat1['F657N_aperture_flux']=cat1['F657_aperture_flux']
cat1['F657N_total_err']=cat1['F657_total_err']
cat1['F657N_aperture_flux_apcorr']=cat1['F657_aperture_flux_apcorr']

ra_diff_arcsec = np.abs(np.asarray(cat1["ra"], dtype=float) - np.asarray(cat0["raj2000"], dtype=float)) * 3600.0
dec_diff_arcsec = np.abs(np.asarray(cat1["dec"], dtype=float) - np.asarray(cat0["dej2000"], dtype=float)) * 3600.0

print(f"Max |delta RA|  (arcsec): {np.nanmax(ra_diff_arcsec):.6e}")
print(f"Max |delta Dec| (arcsec): {np.nanmax(dec_diff_arcsec):.6e}")
assert np.nanmax(ra_diff_arcsec) < 1.0, "RA difference too large"
assert np.nanmax(dec_diff_arcsec) < 1.0, "Dec difference too large"


ratio=True
bands=["F200W","F300M","F335M","F360M"]
JR_apcorr = [-0.63, -0.68, -0.66, -0.67]
froot = "ngc3351_RI_JR_"
yra_flux=[.2,1.5]
logy_flux = False
xcol=["F200W","F300M"]

bands = ['F657N','F187N', 'F770W', 'F1000W']
JR_apcorr = [-0.078, -0.63, -0.75, -0.75]
bands = ['F657N','F770W', 'F1000W']
JR_apcorr = [-0.078, -0.75, -0.75]
xcol=["F770W","F1000W"]
froot = "ngc3351_RI_JR_2_"
yra_flux=[.03,30]
logy_flux = True

apcorr=False

if apcorr:
    k1='_aperture_flux_apcorr'
else:
    k1='_aperture_flux'


pl.clf()
for iband,band in enumerate(bands):
    pl.subplot(4,1,iband+1)
    x = cat0['flux_'+band] 
    if not apcorr: x *= 10**(JR_apcorr[iband]/2.5)
    dx = cat0['er_flux_'+band]

    if band == 'F657N':
        flux2density = 1e6/6e12 # 1e-20 erg/s/cm2 -> mJy
        cat1[band+k1] *= flux2density
        cat1[band+'_total_err'] *= flux2density

    if ratio:
        y = cat1[band+k1] / x
        dy = np.abs(y) * np.sqrt( (cat1[band+'_total_err'] / cat1[band+k1])**2 + (dx / x)**2 )
    else:
        y = cat1[band+k1]
        dy = cat1[band+'_total_err']

    pl.errorbar(x, y, xerr=dx, yerr=dy, fmt="k.", ms=2, alpha=0.2)
    pl.plot(x, y, "k.", ms=3, alpha=0.7)

    xx = np.array(pl.xlim())
    if ratio:
        medval = np.nanmedian(cat1[band+k1] / x)
        pl.plot(xx, medval * np.ones_like(xx), "r--", lw=1, label=f"{band} median:{medval:.2f}")
    else:
        pl.plot(xx, xx, "r--", lw=1)

    pl.xscale("log")
    pl.xlabel(band+" JR (mJy)")
    if ratio:
        pl.ylabel(band+" RI/JR")
        pl.ylim(yra_flux)
        if logy_flux:
            pl.yscale("log")
    else:
        pl.ylabel(band+" RI (mJy)")
        pl.yscale("log")
    pl.legend(prop={'size': 8})

pl.tight_layout()
pl.gcf().set_size_inches(6, 8)
pl.subplots_adjust(hspace=0.22,top=0.95,bottom=0.1,left=0.15)
if apcorr:
    out_plot  = phot_dir / f"{froot}flux_apcorr.png"
else:
    out_plot  = phot_dir / f"{froot}flux_noapcorr.png"
pl.savefig(out_plot, dpi=200)

print(f"Saved plot: {out_plot}")


pl.clf()
pl.gcf().set_size_inches(6, 6)
pl.subplot(2,1,1)
ycol=2.5*np.log10(cat0['flux_'+xcol[1]]/cat0['flux_'+xcol[0]]).data
pl.plot( cat0['flux_'+xcol[0]], ycol, "k.", ms=3, alpha=0.6, label = "JR")
pl.ylabel(f"{xcol[0]}-{xcol[1]} color")
pl.ylim(-2,0.5)
yy=pl.ylim()
pl.xscale("log")
pl.xlim(1e-4,0.5)
ymed=np.nanmedian(ycol)
yrms=np.nanstd(ycol[np.isfinite(ycol)*(ycol>-2)*(ycol<0.5)]-ymed)
pl.plot(pl.xlim(),[-0.88,-0.88],'k--',alpha=0.5, label="RJ: -0.88")
pl.fill_between(pl.xlim(),[ymed+yrms,ymed+yrms],[ymed-yrms,ymed-yrms],color='grey',alpha=0.2, label=f"RMS:{yrms:.2f}")
pl.plot(pl.xlim(),np.ones(2)*ymed,'r--',alpha=0.5, label=f"median:{ymed:.2f}")
pl.legend(prop={'size': 8})

pl.subplot(2,1,2)
ycol=2.5*np.log10(cat1[xcol[1]+k1]/cat1[xcol[0]+k1]).data
pl.plot( cat1[xcol[0]+k1], ycol, "k.", ms=3, alpha=0.6, label = "RI")
pl.xlabel(f"{xcol[0]} flux (mJy)")
pl.ylabel(f"{xcol[0]}-{xcol[1]} color")
pl.ylim(yy)
pl.xscale("log")
pl.xlim(1e-4,0.5)
ymed=np.nanmedian(ycol)
yrms=np.nanstd(ycol[np.isfinite(ycol)*(ycol>-2)*(ycol<0.5)]-ymed)
pl.plot(pl.xlim(),[-0.88,-0.88],'k--',alpha=0.5,label='RJ: -0.88')
pl.fill_between(pl.xlim(),[ymed+yrms,ymed+yrms],[ymed-yrms,ymed-yrms],color='grey',alpha=0.2, label=f"RMS:{yrms:.2f}")
pl.plot(pl.xlim(),np.ones(2)*ymed,'r--',alpha=0.5, label=f"median:{ymed:.2f}")
pl.legend(prop={'size': 8})

pl.tight_layout()
if apcorr:
    out_file = phot_dir / f"{froot}color_apcorr.png"
else:
    out_file = phot_dir / f"{froot}color_noapcorr.png"
pl.savefig(out_file, dpi=200)

print(f"Saved plot: {out_file}")