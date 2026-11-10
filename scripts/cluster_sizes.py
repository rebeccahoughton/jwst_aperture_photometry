#!/usr/bin/env python3
"""examine cluster size as a function of flux and band"""

from pathlib import Path

import matplotlib.pyplot as pl
pl.ion()
import numpy as np
from astropy.table import Table
from astropy import units as u
from scipy import odr


bands = ["F200W","F300M","F335M",'convolved_pah_f360m']

phot_dir = Path("/Users/ri3e/cv/galaxies/phot/from_hst/")
cat_path = phot_dir / "ngc3351_combined.ecsv"

cat = Table.read(cat_path)

nband=len(bands)
pl.clf()
for iband,band in enumerate(bands):
    pl.subplot(nband,1,iband+1)
    wid = cat[f'{band}_psffit_ampwid_wout']/4 * cat.meta[f'{band}_pixel_scale'] * 2.354 # sigma to FWHM
    flux = cat[f'{band}_psffit_ampwid_flux']

    good=np.where((flux > 1e-5)*(wid > 1e-4*u.arcsec))[0] # resolved sources only
    good = good[wid[good] < wid[good].max()*0.9]
    pl.plot(flux[good], wid[good], ".", ms=2, label=band)

    log_flux = np.log10(np.asarray(flux[good], dtype=float))
    width_arcsec = wid[good].to_value(u.arcsec)
    model = odr.Model(lambda beta, x: beta[0] * x + beta[1])
    fit = odr.ODR(
        odr.RealData(log_flux, width_arcsec),
        model,
        beta0=[0.0, np.median(width_arcsec)],
    ).run()
    fit_log_flux = np.linspace(log_flux.min(), log_flux.max(), 100)
    myplot, =pl.plot(
        10**fit_log_flux,
        model.fcn(fit.beta, fit_log_flux),
        "-",
        label=f"fit: {fit.beta[0]:.3g} log10(flux) + {fit.beta[1]:.3g}",
    )
    
    pl.xscale('log')
    #pl.yscale('log')
    pl.xlim(1e-4,1)
    pl.legend(prop={'size':8})

pl.tight_layout()
pl.gcf().set_size_inches(6,8)
pl.savefig("cluster_size_flux.png")


pl.clf()
bandinfo = Table.read("src/data/filter_data.csv",format="ascii.csv", comment="#")
for iband,band in enumerate(bands):
    wid = cat[f'{band}_psffit_ampwid_wout']/4 * cat.meta[f'{band}_pixel_scale'] * 2.354
    flux = cat[f'{band}_psffit_ampwid_flux']

    good=np.where((flux > 1e-5)*(wid > 1e-4*u.arcsec))[0] # resolved sources only
    good = good[wid[good] < wid[good].max()*0.9]

    h,b = np.histogram(wid[good].value, bins=10,range=[0,0.4],density=True)
    myplot,=pl.plot(b[:-1], h, label=band)
    iband=np.where(band.upper()==bandinfo['filter_name'])[0]
    pl.plot(bandinfo['empirical_fwhm_pix'][iband]*cat.meta[f'{band}_pixel_scale']*np.ones(2), 
            [0,15],":",color=myplot.get_color())

pl.legend(prop={'size':8})
pl.xlabel("Gaussian 'unconvolved' FWHM (arcsec)")

pl.gcf().set_size_inches(6,4)
pl.savefig("cluster_size_hist.png")

print(f"Saved plot: cluster_size_hist.png")
