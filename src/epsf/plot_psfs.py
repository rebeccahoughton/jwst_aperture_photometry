from astropy.io import fits
import matplotlib.pyplot as pl
pl.ion()
import numpy as np
from glob import glob
from utils import radial_profile

band="F200W"

stpsf_dir = "/Users/ri3e/cv/jwst_clusters/instrument/psf/" 
empirical_dir = "/Users/ri3e/cv/galaxies/phot/epsf/sample2/"

stpsf_files = glob(f"{stpsf_dir}/*{band}*.fits")
empirical_files = glob(f"{empirical_dir}/*{band}*oversampled2.fits")

if len(stpsf_files) > 1:
    print("multiple STPSF files found:", stpsf_files)
if len(empirical_files) > 1:
    print("multiple empirical files found:", empirical_files)

stpsf_data = fits.getdata(stpsf_files[0])
empirical_data = fits.getdata(empirical_files[0])
empirical_data /= np.nansum(empirical_data) # the oversampled one is not yet normalized 

kpixscale = ['PIXSCALE','PIXELSCL']
stpsf_header = fits.getheader(stpsf_files[0])
empirical_header = fits.getheader(empirical_files[0])
for k in kpixscale:
    if k in stpsf_header:
        stpsf_pscale = stpsf_header[k]
    if k in empirical_header:
        empirical_pscale = empirical_header[k]

r_stpsf, prof_stpsf = radial_profile(stpsf_data)
r_stpsf *= stpsf_pscale
r_empirical, prof_empirical = radial_profile(empirical_data)
r_empirical *= empirical_pscale

pl.clf()
myplot,=pl.plot(r_stpsf, prof_stpsf, label='STPSF')
yy=pl.ylim()
halfmax = prof_stpsf.max()/2
rhalf = np.interp(halfmax, prof_stpsf[::-1], r_stpsf[::-1])
#z=np.min(np.where(prof_stpsf < halfmax))
pl.plot(rhalf*np.ones(2), [0,halfmax],"--",color=myplot.get_color(),label=f'Half-max radius={rhalf:.3f}')

myplot,=pl.plot(r_empirical, prof_empirical, label='Empirical')
halfmax = prof_empirical.max()/2
rhalf = np.interp(halfmax, prof_empirical[::-1], r_empirical[::-1])
pl.plot(rhalf*np.ones(2), [0,halfmax],"--",color=myplot.get_color(),label=f'Half-max radius={rhalf:.3f}')

pl.xlabel('Radius (arcsec)')
pl.ylabel('Profile')
pl.legend(prop={'size': 10})
pl.ylim(0, np.nanmax(np.concatenate([prof_stpsf.flat, prof_empirical.flat]))*1.05)
pl.xlim(0,0.5)
pl.title(band)
pl.savefig(f"epsf_profile_{band}.png")

print("beh")