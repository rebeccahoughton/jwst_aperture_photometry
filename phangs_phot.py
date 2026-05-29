# This notebook was modified from https://github.com/JaysonAstro/prototype_HST_catalog_photometry/blob/main/HST_cats_with_IRAFStarFinder.ipynb
# which is based on https://qosmicqi.github.io/XRBID/chapters/photometry.html#sec-runphots
# and https://www.astropy.org/ccd-reduction-and-photometry-guide/v/pdev/notebooks/photometry/00.00-Preface.html


import glob
import numpy as np
import matplotlib.pyplot as plt
import tomllib
import os
from sys import exit

import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.table import join
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord, match_coordinates_sky

# Photutils imports
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.detection import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry

# ------------------------------------------------
# Configs
# ------------------------------------------------

config_file = 'config/config.toml'     # Photometry parameters
local_file = 'locals/local.toml'        # Paths to directories

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

# Unpack the parameters from the config file
conf = load_config(config_file)
local = load_config(local_file)

# Get top level parameters
steps   = conf['steps']
targets = conf['targets']
bands   = conf['bands']
projects = conf['projects']
product = conf['product']
version = conf['version']
ptype = conf['ptype']

# Number of galaxies to run for
# TODO: if this is going into pjpost, then this will come from the argument 
# TODO: parser?
num_targets = len(targets)

finder_params = conf['parameters']['source_find']
phot_params = conf['parameters']['photometry']

# exit()

# ------------------------------------------------
# Conversions and file management
# ------------------------------------------------
def get_path_to_file(wdir, version, project, galaxy, ptype):
     """Get the path to the data file based on the version, project, galaxy, product type, and filter.
     Args:
          version: version of the data (e.g., v4p1)
          project: JWST PID (e.g., 4793)
          galaxy: galaxy name 
          ptype: product type (e.g., images (for anchored), features, psfmatch, etc.)
          filter: filter name."""
     # TODO: Add functionality for files not in the release directory
     path = f"{wdir}{version}/{project}/release/{galaxy}/{ptype}/"

     # Check that the path exists
     if os.path.exists(path):
          print(f"Found file for {galaxy} {filter} in {path}")
     else:
          raise FileNotFoundError(f"No file found for {galaxy} {filter} in {path}. Please check the path and file naming conventions.")
     return path



def convert_aperture_sum_Jy_per_sr_to_abmag(aperture_sum_jy_sr, header):
     """Convert aperture sum in Jy/sr to AB magnitudes.
     Args:
          aperture_sum_jy_sr: aperture sum in Jy/sr or MJy/sr
          header: FITS header containing WCS information to get pixel area in steradians
                  and BUNIT for checking units of the input aperture sum.
     Returns:
          AB magnitudes"""
     
     # Check that the input is in Jy/sr
     if header.get('BUNIT', '').lower() in ['mjysr', 'mjy/sr', 'mj/steradian']:
          # If header is in MJy/sr, then convert to Jy/sr before calculating magnitude
          print(f"Warning: BUNIT in header is {header.get('BUNIT', 'unknown')}, but expected Jy/sr. Applying conversion to MJy/sr.")
          aperture_sum_jy_sr = np.array(aperture_sum_jy_sr) * 1e6
     elif not header.get('BUNIT', '').lower() in ['jy/sr', 'jy/steradian']:
          raise ValueError("Input aperture sum must be in Jy/sr or MJy/sr for conversion to AB magnitudes.")
    
     # Get pixel area in steradians from header
     pix_area_sr = get_pixarea_in_sr(header)
     fnu_jy = np.array(aperture_sum_jy_sr) * pix_area_sr
     fnu_jy = np.where(fnu_jy > 0, fnu_jy, np.nan)
     # Converrt to magnitudes
     abmag = -2.5 * np.log10(fnu_jy / 3631.0)
     return abmag



def get_pixarea_in_sr(header):
    """Get pixel area in steradians from FITS header.
    Args:
        header: FITS header containing WCS information
    Returns:
        pixel area in steradians (float)"""
    
    # JWST data should have a PIXAR_SR keyword
    if 'PIXAR_SR' in header:
        return float(header['PIXAR_SR'])
    
    # If keyword is not found, then we can try to compute it from CDELT or CD matrix
    elif ('CDELT1' in header) and ('CDELT2' in header):
        print("Warning: PIXAR_SR keyword not found in header. Computing pixel area from WCS information.")
        area_deg2 = np.abs(float(header['CDELT1']) * float(header['CDELT2']))
        if np.isfinite(area_deg2) and (area_deg2 > 0):
            return float((area_deg2 * u.deg**2).to(u.sr).value)
    elif 'CD1_1' in header:
        print("Warning: PIXAR_SR keyword not found in header. Computing pixel area from WCS information.")
        cd = np.array([[float(header['CD1_1']), float(header['CD1_2'])],
                        [float(header['CD2_1']), float(header['CD2_2'])]])
        area_deg2 = np.abs(np.linalg.det(cd))
        return float((area_deg2 * u.deg**2).to(u.sr).value)
    # And if we can't do either of those...
    else:
        raise ValueError("could not get pixel area in steradians from header/WCS")
     


def open_jwst(path, gal, dir, band, mosaic_ext="*anchor*.fits", get_coverage=True):
     """
     Open JWST data (from either MIRI/NIRCam) and return image, error, header.
     Using the stage 3 aligned data products, and it defaults to the anchored mosaic (which is the most aligned product).

     Args:
          path: path to the data directory
          gal: galaxy name
          dir: directory name
          band: filter name (e.g., F770W, F1000W, etc.)
          level: useful if data is hidden in a subdirectory (typical for pjpipe outputs)
          mosaic_ext: extension to search for (default is the anchored mosaic)
          get_coverage: whether to return a coverage mask (default True)
     Returns:
          img: 2D array of the image data
          err: 2D array of the error data
          snr_map: 2D array of the signal-to-noise ratio (img/err)
          coverage_mask: 2D boolean array where True indicates no coverage (NaN or zero in img or err)
          header: FITS header of the image data
     """
     # Load the files
     print(f"Searching in {path} for {band} data, with extension: {mosaic_ext}")
     #     files = glob.glob(f"{path}/{gal.lower()}*{band.lower()}*{mosaic_ext}")
     files = glob.glob(path + f"*{band.lower()}*{mosaic_ext}*")
     print(f"Files found: {files}")

     # Sanity check that we are getting only one aligned mosaic
     if len(files) == 0:
          raise FileNotFoundError(f"No files found for {band} in {dir}{gal}")
     elif len(files) > 1:
          print(f"Warning: Multiple files found for {band} in {dir}{gal}. Using the first one: {files[0]}")

     # Initialize variables
     img_file = None
     err_file = None

     # Open the file and use extensions to assign data and header
     with fits.open(files[0]) as hdul:
          img_file = hdul['SCI']
          img = img_file.data
          header = img_file.header
          # Error
          err_file = hdul['ERR']
          err = err_file.data
          err_header = err_file.header
     # Check the names of the image and error extensions 
     print(f"Image file: {img_file}")
     print(f"Error file: {err_file}")

     # Handle NaNs and zeros
     snr_map = np.full_like(img, np.nan)
     valid = (np.isfinite(img)) & (np.isfinite(err)) & (err > 0)
     snr_map[valid] = img[valid] / err[valid]

     # Coverage mask
     if get_coverage:
          coverage_mask = (~np.isfinite(img)) | (img == 0) | (err == 0)
     else:
          coverage_mask = None

     return img, err, snr_map, coverage_mask, header


# ------------------------------------------------
# Background subtraction
# ------------------------------------------------
def subtract_bkg(img, 
          sigma=3.0, 
          box_size=50, 
          filter_size=3, 
          bkg_estimator=MedianBackground(), 
          coverage_mask=False):
     """Estimate and subtract background from image using Background2D.
     Args:
          img: 2D array of image data
          box_size: size of boxes for background estimation (in pixels)
          filter_size: size of median filter to apply to background (in pixels)
          bkg_estimator: background estimator to use (default is MedianBackground())
          coverage_mask: boolean array where True indicates pixels to exclude from background estimation 
                         (e.g., due to low coverage or bad data). If False, no mask is applied.
     Returns:
          img_sub: background-subtracted image
          bkg_mean: mean background level (in same units as img)
          bkg_rms: background RMS (in same units as img)"""
     
     # estimate background
     # TODO: need to include valid mask based on weight image or other metric
     sigma_clip = SigmaClip(sigma=sigma)
     filter_size = (filter_size, filter_size)
     box_size = (box_size, box_size)
     bkg_estimator = eval(bkg_estimator) if isinstance(bkg_estimator, str) else bkg_estimator

     if coverage_mask is False:
          print(f"Creating coverage mask")
          coverage_mask = (~np.isfinite(img)) | (img == 0)

     bkg = Background2D(
          img,
          box_size=box_size,
          filter_size=filter_size,
          sigma_clip=sigma_clip,
          bkg_estimator=bkg_estimator,
          coverage_mask=coverage_mask,
          )

     rms_map = np.array(bkg.background_rms, dtype=float)
     valid_rms = (~coverage_mask) & np.isfinite(rms_map) & (rms_map > 0)

     #print(f"bkg array {bkg.background}")
     bkg_rms = np.nanmedian(rms_map[valid_rms]) if np.any(valid_rms) else np.nan
     bkg_mean = np.nanmean(np.asarray(bkg.background, dtype=float)[~coverage_mask])
     print(f"Mean background: {bkg_mean}")
     print(f"Background rms: {bkg_rms}")
     print(f"Subtracting background...")
     img_sub = img - bkg.background
     print(f"Background subtraction complete.")

     return img_sub, bkg_mean, bkg_rms, bkg.background


# ------------------------------------------------
# Source finding (using IRAF, DAO in progress)
# ------------------------------------------------
def run_source_finder(img, header, rms, finder='iraf', snr_threshold=3.0, fwhm=2.0, box_size=(50,50),
     roundlo=-0.5, roundhi=0.5, sharplo=0.2, sharphi=1.0, nsources=10000):
     """Find sources in the image using IRAFStarFinder.
     Args:
          img: 2D array of background-subtracted image data
          header: FITS header of the image (used for WCS and pixel scale)
          rms: background RMS (in same units as img)
          finder: source finder to use (currently only 'iraf' supported)
          snr_threshold: signal-to-noise ratio threshold for source detection
          fwhm: FWHM of the PSF in pixels (used for source detection)
          roundlo, roundhi: roundness limits for source selection
          sharplo, sharphi: sharpness limits for source selection
          brightest: if not None, only return this many brightest sources in the catalog
     Returns:
          sources: Table of detected sources with columns xcentroid, ycentroid, flux, sharpness, roundness, mag, peak, etc."""
     # Run the source finder
     print(f"Running source finder: {finder}")
     ths = snr_threshold * rms
     if finder == 'iraf':
          # IRAF always uses circular Gaussian kernels and calculates object's centroid, roundness,
          # and sharpness using imagemoments. 
          source_finder = IRAFStarFinder(threshold=ths,
               fwhm=fwhm,
               roundlo=roundlo,
               roundhi=roundhi,
               sharplo=sharplo,
               sharphi=sharphi,
               brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)
     elif finder == 'dao':
          # DAO can also use elliptical apertures.
          source_finder = DAOStarFinder(threshold=ths,
               fwhm=fwhm,
               roundlo=roundlo,
               roundhi=roundhi,
               sharplo=sharplo,
               sharphi=sharphi,
               brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)
     if finder == 'peaks':
          # find_peaks looks for local maxima above a specified threshold.
          # Requires a bit of extra work to get results in the same format as IRAFStarFinder/DAOStarFinder, 
          # and it doesn't calculate sharpness or roundness.
          # TODO: Add function converting find_peaks output to a table with xcentroid, ycentroid, flux, etc.
          # Check if box size is even. If it is, add one to each of the values
          if box_size[0] % 2 == 0:
               box_size = (box_size[0] + 1, box_size[1] + 1)
          sources = find_peaks(img, 
               threshold=ths, 
               box_size=box_size,
               centroid_func=centroid_quadratic,
          )
     elif finder != 'iraf' and finder != 'dao' and finder != 'peaks':
          raise ValueError(f"Starfinder {finder} not recognized. Currently only 'iraf' supported.")

     
     print(f"Found {len(sources)} sources")
     print(sources.colnames)
     return sources


# TODO: these things
def load_source_catalog():
     print("Load an external source catalog to use for photometry.")

     
def filter_catalog():
     print("Filtering catalog based on morphology and other criteria.")


# ------------------------------------------------
# Optimal aperture and photometry
# ------------------------------------------------
def get_optimal_aperture(data, sources, max_r=32, brightest=50, frac=0.95, plot=True):
     """Find the optimal aperture radius to use for the photometry from the 
        curve of growth of the brightest n sources. 
     Args:
          data: 2D array of image data (background-subtracted)
          sources: Table of sources from source finder 
                   (must contain xcentroid, ycentroid, flux)
          max_r: maximum aperture radius to test (in pixels)
          brightest: if not None, only use this many brightest sources to compute curve of growth
          frac: fraction of total flux to use as criterion for optimal radius 
          (e.g., 0.95 means radius where median curve of growth reaches 95% of total flux)
          plot: if True, plot the curve of growth and optimal radius
          
     Returns:
          r_opt: optimal aperture radius in pixels (to use with compute_photometry)"""
     
     # Select only the brightest sources to compute the curve of growth
     if brightest is not None:
          sources = sources[np.argsort(sources['flux'])[-brightest:]]
          print(f"Using only {len(sources)} sources.")

     print("Doing aperture photometry...")
     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
     radii = np.arange(1, max_r)

     # Define in and outer annuli for local background estimation
     # TODO: optimize values for the sky annulus 
     ann_in, ann_out = max_r + 2, max_r + 8
     ann = CircularAnnulus(positions, r_in=ann_in, r_out=ann_out)

     # Get local backgrounds
     # TODO: consider impact of extended emission on local background. 
     ann_phot = aperture_photometry(data, ann)
     bkg_mean = np.asarray(ann_phot["aperture_sum"]) / ann.area

     # At each radius, compute photometry
     fluxes = []
     for r in radii:
          ap = CircularAperture(positions, r=r)
          phot = aperture_photometry(data, ap)

          # Subtract local background
          src = np.asarray(phot["aperture_sum"]) - bkg_mean * ap.area
          fluxes.append(src)

     # Normalize fluxes for computing the curve of growth
     fluxes = np.asarray(fluxes).T

     norm = fluxes / fluxes[:, [-1]]
     norm[~np.isfinite(norm)] = np.nan

     # comptue median normalized flux
     median_curve = np.nanmedian(norm, axis=0)  
     # Get the index of the radius where the curve of growth reaches the specified fraction of total flux
     idx = np.where(median_curve >= frac)[0]
     r_opt = radii[idx[0]] if len(idx) else radii[np.nanargmax(median_curve)]
     print(f"Optimal aperture radius: {r_opt}")

     if plot:
          plt.figure()
          plt.plot(radii, median_curve, marker='o')
          plt.axvline(r_opt, color='red')
          plt.xlabel("Aperture radius (pixels)")
          plt.ylabel("Normalized flux")
          plt.title("Curve of growth")
          plt.grid(True)

     return r_opt


# ------- Main photometry function ------------------------------------------------ 
def compute_photometry(data, 
          header, 
          sources, 
          gal, 
          band,
          aperture_radius=10, 
          radius_sky_in=12, 
          radius_sky_out=18, 
          use_brightest=False, 
          write=False, 
          outdir='./'):
     """Compute aperture photometry for sources and return catalog with RA, Dec, magnitudes, etc.
     
     Args:
          data: 2D array of image data (background-subtracted)
          header: FITS header of the image
          sources: Table of sources from source finder 
                   (must contain xcentroid, ycentroid, flux, sharpness, roundness, mag, peak)
          aperture_radius: radius of circular aperture to use for photometry (in pixels)
          radius_sky_in: inner radius of the sky annulus (in pixels)
          radius_sky_out: outer radius of the sky annulus (in pixels)
          use_brightest: if True, only use the brightest sources for photometry
          write: if True, write catalog to outdir with name {gal}_jwst_{band}_cat.fits
          outdir: directory to write catalog if write=True

     Returns:
          phot_full: Table with photometry results, including RA, Dec, aperture sum, magnitudes, etc.
     """

     if use_brightest is not False:
          # Aperture photometry of only brightest sources
          sources = sources[np.argsort(sources['flux'])[-use_brightest:]]
          print(f"using only {len(sources)} sources")

     # Do aperture photometry
     print(f"Doing aperture photometry...")
     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
     apertures = CircularAperture(positions, r=aperture_radius)
     aper_stats = ApertureStats(data, apertures)
     phot_full = aperture_photometry(data, apertures, method='exact')

     # Annulus
     annuli = CircularAnnulus(positions, r_in=radius_sky_in, r_out=radius_sky_out)
     sigma_clip_bkg = SigmaClip(sigma=3.0, maxiters=5)
     # mask = annuli.to_mask(method='exact')
     # Mask the data to exclude NaNs and infs from the background estimation
     mask = ((np.isinf(data)) | (np.isnan(data)))

     # Background annulus stats
     bkg_stats = ApertureStats(data, annuli, sigma_clip=sigma_clip_bkg, mask=mask, sum_method='exact')
     bkg_median = bkg_stats.median
     bkg_median[np.isnan(bkg_median)]=0
     area_aper = aper_stats.sum_aper_area.value
     area_annulus = bkg_stats.sum_aper_area.value
     total_bkg = bkg_median * area_aper

     # Subtract background from aperture sum
     phot_full['aperture_sum'] = phot_full['aperture_sum'] - total_bkg

     # Copy source-finder morphology columns
     phot_full['flux'] = np.asarray(sources['flux'])
     phot_full['sharpness'] = np.asarray(sources['sharpness'])
     if 'roundness' in sources.colnames:
          phot_full['roundness'] = np.asarray(sources['roundness'])          
     phot_full['finder_mag'] = np.asarray(sources['mag'])
     phot_full['peak'] = np.asarray(sources['peak'])

     # Include ra, dec
     wcs = WCS(header)
     ra, dec = wcs.all_pix2world(phot_full["xcenter"], phot_full["ycenter"], 0)
     phot_full["ra"] = ra
     phot_full["dec"] = dec

     # Convert flux from the source finder in table (converted to AB magnitudes)
     phot_full['finder_flux_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(sources['flux'], header=header)
     # Aperture sum from circular aperture photometry (converted to AB magnitudes)
     phot_full['aperture_sum_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header)

     # Sort by aperture flux
     phot_full.sort("aperture_sum")
     phot_full.reverse()

     # Print the column names of the photometry table
     print(phot_full.colnames)

     # Write the catalog if requested
     if write:
          cat_name = f"{gal}_jwst_{band}_cat.fits"
          print(f"Writing catalog to {outdir + cat_name}")
          phot_full.write(outdir + cat_name, overwrite=True)

     return apertures, phot_full


# ------------------------------------------------
# Other useful functions
# ------------------------------------------------

# Load in the catalogs that are produced by the image3pipeline
def get_image3_catalog(filedir, filter, galaxy, level='lv3'):
    cat_dir = filedir
    # cat_dir = dir + f"{galaxy}/{filter.upper()}/{level}"
    cat_filename = f"{galaxy}_nircam_{level}_{filter.lower()}_cat_align.ecsv"
    cat_name = cat_dir + "/" + cat_filename
    return cat_name


# Cross match the catalog that we have made with the outputs of the image3pipeline
def cross_match_catalogs(dir, filter, galaxy, phot_full, cat_image3):
    cat_name = get_image3_catalog(dir, filter, galaxy=galaxy)
    calib_cat = Table.read(cat_name, format='ascii.ecsv')

    # Use proximity based approach to cross match the catalogs
    calib_coords = SkyCoord(ra=calib_cat['ra'] * u.deg, dec=calib_cat['dec'] * u.deg)
    # My photometry into Sky Coords
    phot_coords = SkyCoord(ra=phot_full['ra'] * u.deg, dec=phot_full['dec'] * u.deg)
    # Match coordinates
    ind_2d_cat, dist_2d, _ = match_coordinates_sky(phot_coords, calib_coords)
    return ind_2d_cat, dist_2d, phot_full



# Empirical filter FWHM
# NIRCAM from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions#gsc.tab=0
# MIRI from https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions#gsc.tab=0
filter_fwhm = {
    'F150W': 1.613,
    'F164N': 1.806,
    'F187N': 2.065,
    'F200W': 2.129,
    'F212N': 2.323,
    'F277W': 1.460,
    'F300M': 1.587,
    'F335M': 1.762,
    'F360M': 1.905,
    'F405N': 2.159,
    'F444W': 2.302,
    'F770W': 2.445,
    'F1000W': 2.982,
    'F1130W': 3.409,
    'F2100W': 6.127,
}



# Directories
jwst_dir = local['jwst_dir']
out_dir = local['out_dir']

# Check that outdir exists
if not os.path.exists(out_dir):
     raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
     exit()

# Determine the path based on the
path = get_path_to_file(
     wdir=jwst_dir, 
     version=version, 
     project=projects[0], 
     galaxy=targets[0],
     ptype=ptype[0])

# This is only still here temporarily
use_filter_fwhm = True 

def do_photometry(
          steps, 
          targets,
          use_filter_fwhm,
          conf,
     ):
     """Main function to run the photometry steps for each galaxy and filter.
     Args:
          steps: list of steps to run (e.g., ['bkg_subtract', 'source_find', 'r_opt', 'photometry'])
          targets: list of galaxy names to process
          use_filter_fwhm: this will eventually go into the config
          conf: dictionary of parameters from the config file."""

     print(" ")
     for gal in targets:
          for band in bands:
               print(f"Processing {gal} at {band}...")

               # Open the JWST data file 
               img, err, snr_map, coverage_mask, header = open_jwst(
                    path=path, 
                    gal=gal, 
                    dir=jwst_dir, 
                    band=band
               )

               # Subtract background 
               img_sub, bkg_mean, bkg_rms, bkg_background = subtract_bkg(
                    img=img, 
                    **conf['parameters']['bkg_subtract'],
               )

               if 'source_find' in steps:
                    # Get sources using the source finder
                    sources = run_source_finder(
                         img=img_sub, 
                         header=header, 
                         rms=bkg_rms, 
                         **conf['parameters']['source_find'],
                    )

               # **** Alternatively, load in a catalog computed by another method here ****
               # TODO: if loading in another catalog, need a path to it in local.toml. 

               # Either get the optimum radius based on curve of growth...
               if 'r_opt' in steps:
                    print(f"Computing optimal aperture for photometry...")
                    r_opt = get_optimal_aperture(
                         data = img_sub, 
                         sources = sources,
                         **conf['parameters']['r_opt']
                    )
               else:
                    r_opt = conf['parameters']['photometry']['aperture_radius']

               # Update the fwhm according to the filter if use_filter_fwhm is True.
               # If use_filter_fwhm is False, stay at specified value.
               if use_filter_fwhm:
                    try:
                         fwhm = filter_fwhm[band.upper()]
                         print(f"Using FWHM of {fwhm} pixels for source detection based on JWST PSF for {band.upper()}.")
                    except KeyError:
                         print(f"Warning: FWHM for {band.upper()} not found in filter_fwhm dictionary. Using default FWHM of {fwhm} pixels for source detection.")

               # ...or just set it to a fixed value (e.g., based on the PSF FWHM)
               print(f"Setting aperture radius to {r_opt} pixels.")
               # Check r_opt relative to the FWHM of the filter:
               if r_opt > 3 * fwhm:
                    print("Large r_opt. Using PSF FWHM rather than curve of growth for photometry.")
                    r_opt = 2.5 * fwhm

               # Perform photometry with circular apertures
               if 'aperture_photometry' in steps:
                    print(f"Performing photometry on {len(sources)} sources with aperture radius of {r_opt} pixels.")
                    apertures, catalog = compute_photometry(
                         data = img_sub, 
                         header = header, 
                         sources = sources,
                         **conf['parameters']['photometry']
                    )

                    print(f"Photometry complete. Catalog has {len(catalog)} sources.")

# # Plot the image, background, and background-subtracted image
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# norm = ImageNormalize(vmin=np.nanpercentile(img, 25.00), 
#                       vmax=np.nanpercentile(img, 99.99), 
#                       stretch=LogStretch())
# ax[0].imshow(img, origin='lower', cmap='inferno', norm=norm)
# ax[0].set_title(f"{gal.upper()} {band.upper()} mosaic")
# ax[1].imshow(bkg_background, origin='lower', cmap='inferno')
# ax[1].set_title(f"Estimated background")
# norm_sub = ImageNormalize(vmin=np.nanpercentile(img_sub, 25.00), 
#                           vmax=np.nanpercentile(img_sub, 99.99), 
#                           stretch=LogStretch())
# ax[2].imshow(img_sub, origin='lower', cmap='inferno', norm=norm_sub)
# ax[2].set_title(f"Background-subtracted image")
# # Add colourbars
# for a in ax:
#      im = a.images[0]
#      plt.colorbar(im, ax=a, pad=0.01, fraction=0.05)
# plt.tight_layout()

do_photometry(
     steps=steps, 
     targets=targets, 
     use_filter_fwhm=use_filter_fwhm,
     conf=conf
)
exit()


# ---------------------------------------------------------------------------------------------------------

# Alternative approach using standardised aperture corrections from the JWST CRDS.
path_to_crds = "/nexus/posix0/MIA-astro-env/eschinner/jgonzalez/jwst_pipeline/crds_cache/jwst_ops/references/jwst/" + 'nircam' + "/"

# Get the apcorr file using glob
apcorr_files = glob.glob(path_to_crds + f"*apcorr*")
if len(apcorr_files) == 0:
    raise FileNotFoundError(f"No apcorr files found for {band} in {inst} at {path_to_crds}")
else:
    print(f"Found apcorr files: {apcorr_files}")

# Load the file
apcorr_data = fits.getdata(apcorr_files[0], ext=1)
print(f"APCORR data columns: {apcorr_data.columns.names}")

# Print all the unique filters in the apcorr file
print("Unique eefraction values:", np.unique(apcorr_data['eefraction']))

# Get data for a specific eefraction
# The eefraction is the fraction of the total flux that is enclosed within the aperture radius.
eefraction_value = 0.70
row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]

# Limit to a specific filter 
row = row[(row['filter'] == band.upper())]

# Extract values
wcs_apcorr = WCS(header)
radius = row['radius'][0]   # in pixels
sky_in = row['skyin'][0]    # in pixels
sky_out = row['skyout'][0]  # in pixels
apcorr = row['apcorr'][0]   # factor to multiply enclosed flux to get total flux
print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels and eefraction {eefraction_value}")

# Create apertures for aperture correction, not using the curve of growth
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

# Redo the aperture photometry using the radius, sky_in, and sky_out from the apcorr file
aperture = CircularAperture(positions, r=radius)
sky_annulus = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)
phot_table_apcorr = aperture_photometry(img_sub, aperture, wcs=wcs_apcorr, method='exact')
sky_table_apcorr = aperture_photometry(img_sub, sky_annulus, wcs=wcs_apcorr, method='exact')

# Extract flux and sky
fluxes = phot_table_apcorr['aperture_sum'].value  # MJy/sr
sky_mean = sky_table_apcorr['aperture_sum'].value / sky_annulus.area  
sky_total = sky_mean * aperture.area  

# Correct for sky
net_fluxes = fluxes - sky_total  

# Apply aperture correction 
total_fluxes = net_fluxes * apcorr  

# Convert to AB magnitudes using the pixel area in steradians from the header
pixarea_sr = header['PIXAR_SR']  # in steradians
# total_flux_jy = total_fluxes * 1e6  # MJy → Jy
abmag_apcorr = convert_aperture_sum_Jy_per_sr_to_abmag(total_fluxes, header=header)

# Add column to the phot_table_apcorr with the aperture-corrected AB magnitudes
phot_table_apcorr['ABmag_apcorr'] = abmag_apcorr

# Create merged table on the x and y coordinates of the sources to compare 
# the aperture-corrected magnitudes with the original photometry catalog
merged_table = join(phot_table_apcorr, catalog, keys=['xcenter', 'ycenter'])


# Make a histogram of the aperture sums in the catalog
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(catalog['aperture_sum_abmag'], bins=30, alpha=0.5, label='Original photometry')
ax.hist(phot_table_apcorr['ABmag_apcorr'], bins=30, alpha=0.5, label='Aperture-corrected')
ax.set_xlabel('Aperture Sum')
ax.set_ylabel('Frequency')
ax.legend()
plt.show()


