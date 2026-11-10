import os
import fnmatch
import tomllib
import numpy as np
from astropy.io import fits
import glob
from astropy import units as u
from astropy.stats import SigmaClip
from astropy.table import Table, hstack

from scipy.spatial import cKDTree
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.utils import calc_total_error



def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

    

def load_filter_data(csv_path=None):
    """Load empirical filter metadata from an external CSV file."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "filter_data.csv")

    table = Table.read(csv_path, format="ascii.csv", comment="#")
    fwhm = {}
    wave = {}
    for row in table:
        filter_name = str(row['filter_name']).strip()
        fwhm[filter_name] = float(row['empirical_fwhm_pix'])
        wave[filter_name] = float(row['central_wave'])
    return fwhm, wave


filter_fwhm_pix, filter_wave = load_filter_data()



def get_file(wdir, version, project, galaxy, ptype, filter):
    """Get the data file with full path based on the version, project, galaxy, product type, and filter.
    Args:
        wdir: root of working directory
        version: version of the data (e.g., v4p1)
        project: JWST PID (e.g., 4793)
        galaxy: galaxy name 
        ptype: product type (e.g., images (for anchored), features, psfmatch, etc.)
        filter: filter name."""

    # start in release directory structure, then broaden
    path = f"{wdir}{version}/{project}/release/{galaxy}/{ptype}/"

    # TODO implement better fallbacks starting with mosaic_ext (default anchored) and then falling back to i2d or types
    print(f"Searching in {path} for {filter} data, with extension: {ptype}")
    files = glob.glob(path + f"*{filter.lower()}*")
    if len(files) > 1:
        print(f"Warning: Multiple files found for {filter} in {path}. Using the first one.")
        print(f"Files found: {files}")
        return(files[0])
    elif len(files) == 1:
        print(f"Found file for {filter} in {path}: {files[0]}")
        return(files[0])
    else:
        print(f"No files found for {filter} in {path}. Trying a broader search...")

        # look for plausible files in the directory and print a warning if we find any, but raise an error if we don't find any
        plausible_files = []
        for root, dirnames, filenames in os.walk(wdir + version + "/"):
            for f in fnmatch.filter(filenames, '*.fits'):
                if galaxy.lower() in f.lower() and \
                    ptype.lower() in f.lower() and \
                    filter.lower() in f.lower() and \
                    "resid" not in f.lower() and \
                    "background" not in f.lower() and \
                    "bgsub" not in f.lower() and \
                    "model" not in f.lower() and \
                    ("pah" in filter.lower()) == ("pah" in f.lower()):
                    plausible_files.append(os.path.join(root, f))

        if len(plausible_files) > 1:
            raise RuntimeError(f"Multiple plausible files found for {galaxy} {filter} in release {path}: {plausible_files}")
        elif len(plausible_files) == 1:
            print(f"Found plausible file for {galaxy} {filter} in release {path}: {plausible_files[0]}")
            return plausible_files[0]
        else:
            raise FileNotFoundError(f"No file found for {galaxy} {filter} in {wdir + version + '/'} . Please check the path and file naming conventions.")



def open_jwst(filename, get_coverage=True, extensions_to_try=['SCI','CONTSUB','PRIMARY']):
    """
    Open data and return image, error, header.
    Using the stage 3 aligned data products, and it defaults to the anchored mosaic (which is the most aligned product).

    Args:
         filename: path to the FITS file
         get_coverage: whether to return a coverage mask (default True)
    Returns:
         img: 2D array of the image data
         err: 2D array of the error data
         snr_map: 2D array of the signal-to-noise ratio (img/err)
         coverage_mask: 2D boolean array where True indicates no coverage (NaN or zero in img or err)
         header: FITS header of the image data
    """
    # Load the files
    # Initialize variables
    img_file = None
    err_file = None

    # Open the file and use extensions to assign data and header
    hdul=fits.open(filename)
    #with fits.open(filename) as hdul:
    for ext in extensions_to_try:
         if ext in hdul:
              img_file = hdul[ext]
              header = img_file.header
              img = img_file.data * u.Unit(header['BUNIT'])
              # special case nonstandard HST Ha:
              break
    # Error
    hdunames = [hdu.name for hdu in hdul]
    if 'ERR' in hdunames:
        err_file = hdul['ERR']
        err = err_file.data * u.Unit(header['BUNIT'])
    else:
        # estimate error from image - parameters are from Jimena's HST phot, probably need tuning for JWST
        sigma_clip = SigmaClip(sigma=5., maxiters=10)
        bkg_estimator = SExtractorBackground()
        coverage_mask = (~np.isfinite(img)) | (img == 0)
        bkg = Background2D(
            img,
            (30,30),
            filter_size=(3, 3),
            fill_value=0.0,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            coverage_mask=coverage_mask,
        )
        # 3rd parameter = Ratio of counts (e.g., electrons or photons) to the data units         
        # TODO double check gain units, and also make it work for other instruments besides JWST
        if img.unit.is_equivalent(u.MJy / u.sr):
            # workaround for numpy bug with units fixed but need python 3.15 I think
            err = np.asarray(calc_total_error(img.value, bkg.background.value, 
                              effective_gain = header['XPOSURE']/header['PHOTMJSR'] )) * img.unit
        else:
            print("WARNING: Image unit is not MJy/sr, using effective gain=1")
            err = np.asarray(calc_total_error(img.value, bkg.background.value, 
                              effective_gain = 1.0 ) ) * img.unit
        err_file = "Estimated from image"
    hdul.close()

    # Check the names of the image and error extensions 
    print(f"Image file: {img_file}")
    print(f"Error file: {err_file}")

    # Handle NaNs and zeros
    snr_map = np.full_like(img.value, np.nan)
    valid = (np.isfinite(img)) & (np.isfinite(err)) & (err > 0)
    snr_map[valid] = img[valid] / err[valid]

    # Coverage mask
    if get_coverage:
        coverage_mask = (~np.isfinite(img)) | (img == 0) | (err == 0)
    else:
        coverage_mask = None

    return img, err, snr_map, coverage_mask, header



def get_psf_file(band, psf_dir):
    candidates = glob.glob(psf_dir + f"/PSF*{band}*fits")

    if len(candidates) == 0:
        raise FileNotFoundError(f"No PSF files found for {band} in {psf_dir}")
    elif len(candidates) > 1:
        print(f"Multiple PSF files found for {band} in {psf_dir}.")
        cand2=np.array([x for x in candidates if 'gauss' not in x]) 
        if len(cand2) > 0:
            print(f"Using the first non-gauss one: {cand2[0]}")
            return cand2[0]
        else:
            print(f"Using the first one: {candidates[0]}")
    return candidates[0]  