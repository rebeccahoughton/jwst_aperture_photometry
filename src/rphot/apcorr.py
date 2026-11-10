import glob, os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from astroquery.svo_fps import SvoFps

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry

from .io import load_filter_data, get_psf_file


filter_fwhm_pix, filter_wave = load_filter_data()


def get_apcorr_params(crds_dir, band, inst, eefraction_value=0.8, apcorr_method='crds'):
    """Get the aperture correction parameters from the CRDS apcorr file for a given filter and eefraction.
    Args:
        crds_dir: directory where the CRDS apcorr files are stored
        band: the filter
        eefraction_value: the fraction of total flux enclosed within the aperture.
    Returns:
        radius: the aperture radius in pixels
        sky_in: the inner sky annulus radius in pixels
        sky_out: the outer sky annulus radius in pixels
        apcorr: the aperture correction factor
    """
    # TODO: add option to use multiple methods, each of which ends up with its own column. 

    # Aperture correction for point sources based on the encircled energy fraction
    # from the crds calibration files. Multiply the flux by apcorr.
    if apcorr_method == 'crds':

        print(f"Getting parameters from CRDS for {band} with eefraction {eefraction_value}...")
        apcorr_files = glob.glob(crds_dir + f"*apcorr*")

        # Check that the files exist
        if len(apcorr_files) == 0:
            raise FileNotFoundError(f"No apcorr files found for {band} at {crds_dir}")
        else:
            print(f"Found apcorr files: {apcorr_files} in {crds_dir}")

        # Load the most recent apcorr file (final in the list)
        apcorr_data = fits.getdata(apcorr_files[-1], ext=1)

        # Get data for a specified eefraction and filter
        row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]
        row = row[(row['filter'] == band.upper())]

        if len(row) == 0:
            row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]
            row = row[(row['pupil'] == band.upper())]

        # Extract values
        radius  = row['radius'][0]   # in pixels
        sky_in  = row['skyin'][0]    # in pixels
        sky_out = row['skyout'][0]   # in pixels
        apcorr  = row['apcorr'][0] * u.dimensionless_unscaled  # factor to multiply enclosed flux to get total flux

    # Aperture correction using factors derived in Rodriguez et al. 2025, based on Deger et al. (2022).
    elif apcorr_method == 'cluster':

        print(f"Using aperture correction values from Rodriguez et al. 2025")

        # Load parameters from apcorr_rodriguez.ecsv
        if not os.path.exists('apcorr_rodriguez.ecsv'):
            raise FileNotFoundError("apcorr_rodriguez.ecsv not found. Please make sure the file is in the current directory.")
        
        if inst.lower() == 'nircam':
            apcorr_val = Table.read('apcorr_rodriguez.ecsv', format='ascii.ecsv')
            if band.lower() not in apcorr_val['band'].data:
                raise ValueError(
                    f"Filter {band} not found in Rodriguez et al. 2025 apcorr file."
                    "\n Available filters: {apcorr_val['band'].data}"
                    "\n Please chose a different aperture correction method in the config file [i.e. 'crds']."
            )
            
            # WARNING: these radii are only valid for NIRCAM SHORT, not NIRCAM LONG
            radius = 4 
            an_in = 2.
            an_out = 3. 
            sky_in = an_in * radius
            sky_out = an_out * radius
            apcorr_vega_mag = apcorr_val[apcorr_val['band'] == band.lower()]['apcorr']
            
            # Get the zero point from SVO
            filter_info = SvoFps.get_filter_list(facility='JWST', instrument=inst)
            zero_point_vega = filter_info[filter_info['filterID'] == f'JWST/{inst}.{band.upper()}']['ZeroPoint']
            # delta_mag = - 2.5*np.log10(zero_point_vega/3631.0)
            # Because the aperture correction is a flux ratio, mag system doesn't matter
            apcorr_abmag = apcorr_vega_mag
            apcorr = apcorr_abmag * u.mag

        elif inst.lower() == 'miri':
            print("Cluster aperture corrections not computed for MIRI. Use eefraction = 0.5 (50%) with CRDS.")
            apcorr_files = glob.glob(crds_dir + f"*apcorr*")

            # Check that the files exist
            if len(apcorr_files) == 0:
                raise FileNotFoundError(f"No apcorr files found for {band} at {crds_dir}")
            else:
                print(f"Found apcorr files: {apcorr_files} in {crds_dir}")

            # Load the most recent apcorr file (final in the list)
            apcorr_data = fits.getdata(apcorr_files[-1], ext=1)

            # Get data for a specified eefraction and filter
            row = apcorr_data[apcorr_data['eefraction'] == 0.5]
            row = row[(row['filter'] == band.upper())]
            # Extract values
            radius  = row['radius'][0]   # in pixels
            sky_in  = row['skyin'][0]    # in pixels
            sky_out = row['skyout'][0]   # in pixels
            apcorr  = row['apcorr'][0] * u.dimensionless_unscaled 


    # TODO: add method for correction based on curve of growth. 

    # If nothing is recognised, use a simplified approximation. 
    else:
        print(f"Using default (basic) aperture correction parameters for {band} with eefraction {eefraction_value}...")
        radius = filter_fwhm_pix.get(band.upper(), 2.0)  # default to 2 pixels if filter not found
        sky_in = radius + 2
        sky_out = radius + 8
        apcorr = (1.0 / eefraction_value) * u.dimensionless_unscaled # simple correction factor based on eefraction

    return radius, sky_in, sky_out, apcorr



def get_apcorr_from_psf(band, r_ap, r_sky_in, r_sky_out, 
                        psf_dir="./",
                        psf_file=None # psf_file will override psf_dir if provided
                        ):
    """
    Get aperture correction from PSF for a given band and aperture parameters.

    Parameters
    ----------
    band : str
        The band of the observation.
    r_ap : array-like
        Aperture radii in native (not oversampled) pixel units.
    r_sky_in : array-like
        Inner radii of the sky annulus.
    r_sky_out : array-like
        Outer radii of the sky annulus.
    psf_dir : str, optional
        Directory containing PSF files. Default is "./".
    psf_file : str, optional
        Specific PSF file to use. Overrides psf_dir if provided.

    Returns
    -------
    apcor : array-like 
        Aperture correction factors for each set of aperture parameters.
    """
    apcor = np.zeros_like(r_ap, dtype=float)

    # Find the PSF file for this band and load the oversampled PSF image.
    if psf_file is None:
        psf_file = get_psf_file(band, psf_dir)
    psf_data = fits.getdata(psf_file)

    # input values are in pixel units, not oversampled coordinates. They will be multiplied by the oversampling factor later.
    r_ap = np.asarray(r_ap, dtype=float)
    r_sky_in = np.asarray(r_sky_in, dtype=float)
    r_sky_out = np.asarray(r_sky_out, dtype=float)
    r_ap, r_sky_in, r_sky_out = np.broadcast_arrays(r_ap, r_sky_in, r_sky_out)

    y0 = (psf_data.shape[0] - 1) / 2.0
    x0 = (psf_data.shape[1] - 1) / 2.0
    pos = [(x0, y0)]

    psf_sum_ap = np.zeros(r_ap.shape, dtype=float)
    psf_mean_ann = np.zeros(r_ap.shape, dtype=float)
    npix_ap = np.zeros(r_ap.shape, dtype=float)
    npix_ann = np.zeros(r_ap.shape, dtype=float)

    # check the pixel oversampling
    if 'OVERSAMP' in fits.getheader(psf_file):
        oversamp = fits.getheader(psf_file)['OVERSAMP']
    else:
        print("WARNING: OVERSAMP keyword not found in PSF header. Assuming oversampling factor of 4.")
        oversamp = 4
    

    for i in np.ndindex(r_ap.shape):
        aper = CircularAperture(pos, r = oversamp * r_ap[i])
        ann = CircularAnnulus(pos, r_in = oversamp * r_sky_in[i], r_out = oversamp * r_sky_out[i])
        ap_phot = aperture_photometry(psf_data, aper, method='exact')
        ann_phot = aperture_photometry(psf_data, ann, method='exact')

        npix_ap[i] = float(aper.area)
        npix_ann[i] = float(ann.area)
        psf_sum_ap[i] = float(ap_phot['aperture_sum'][0])
        psf_mean_ann[i] = float(ann_phot['aperture_sum'][0]) / npix_ann[i]

    return np.nansum(psf_data) / ( psf_sum_ap - psf_mean_ann * npix_ap ) * u.dimensionless_unscaled