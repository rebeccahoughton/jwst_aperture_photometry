# This notebook was modified from https://github.com/JaysonAstro/prototype_HST_catalog_photometry/blob/main/HST_cats_with_IRAFStarFinder.ipynb
# which is based on https://qosmicqi.github.io/XRBID/chapters/photometry.html#sec-runphots
# and https://www.astropy.org/ccd-reduction-and-photometry-guide/v/pdev/notebooks/photometry/00.00-Preface.html


import fnmatch
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import tomllib
import os
import pdb
import warnings
from sys import exit
from scipy.spatial import cKDTree

import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table, join, hstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization import ImageNormalize, SqrtStretch, LogStretch
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import rotate


# Photutils imports
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.detection import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry
from photutils.utils import calc_total_error

# SVO for aperture correction
from astroquery.svo_fps import SvoFps

# ------------------------------------------------
# Configs
# ------------------------------------------------

config_file = 'config/config_pahsub_force.toml'     # Photometry parameters
local_file = 'config/local.toml'       # Paths to directories

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
cat_filetype = conf['cat_filetype']

# Number of targets to process
num_targets = len(targets)

finder_params = conf['parameters']['source_find']
phot_params = conf['parameters']['photometry']


def load_filter_data(csv_path=None):
     """Load empirical filter metadata from an external CSV file."""
     if csv_path is None:
          csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filter_data.csv")

     table = Table.read(csv_path, format="ascii.csv", comment="#")
     fwhm = {}
     wave = {}
     for row in table:
          filter_name = str(row['filter_name']).strip()
          fwhm[filter_name] = float(row['empirical_fwhm_pix'])
          wave[filter_name] = float(row['central_wave'])
     return fwhm, wave


filter_fwhm_pix, filter_wave = load_filter_data()


# ------------------------------------------------
# Conversions and file management
# ------------------------------------------------
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
     files = glob.glob(path + f"*{filter.lower()}*{ptype}*")
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
     # Convert to magnitudes
     abmag = -2.5 * np.log10(fnu_jy / 3631.0)
     return abmag



def convert_abmag_to_Jy_per_sr(abmag, header, unit='MJy/sr'):
     """Convert AB magnitudes to Jy/sr.
     Args:
          abmag: AB magnitudes
          header: FITS header containing WCS information to get pixel area in steradians
                  and BUNIT for checking units of the output aperture sum.
     Returns:
          aperture sum in Jy/sr (numpy array)"""
     # Get pixel area in steradians from header
     pix_area_sr = get_pixarea_in_sr(header)
     fnu_jy = 3631.0 * 10**(-0.4 * abmag)
     aperture_sum_jy_sr = fnu_jy / pix_area_sr
     if unit == 'MJy/sr' or unit == 'MJ/sr' or unit == 'mjy/sr':
          aperture_sum_jy_sr = aperture_sum_jy_sr * 1e-6
     return aperture_sum_jy_sr



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
     


def open_jwst(filename, get_coverage=True):
     """
     Open JWST data (from either MIRI/NIRCam) and return image, error, header.
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
     with fits.open(filename) as hdul:
          img_file = hdul['SCI']
          img = img_file.data
          header = img_file.header
          # Error
          hdunames = [hdu.name for hdu in hdul]
          if 'ERR' in hdunames:
               err_file = hdul['ERR']
               err = err_file.data
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
               err = calc_total_error(img, bkg.background, effective_gain = header['XPOSURE']/header['PHOTMJSR']) 
               # TODO double check units of calc_total_error() 
               err_file = "Estimated from image"
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


def match(
    catalog1, 
    catalog2, 
    npix=2, 
    keys=['catalog1', 'catalog2']):
    
    coords2 = np.array([catalog2['xcenter'], catalog2['ycenter']]).T
    coords1 = np.array([catalog1['xcenter'], catalog1['ycenter']]).T
    # Build a KD-tree for the first catalog
    tree = cKDTree(coords2)

    # Find matches within npix pixels
    # pixel_scale = 1#0.031
    max_distance = npix  # or e.g. 1.0 pixel if you want 1 pixel tolerance

    distances, indices = tree.query(coords1, k=1, distance_upper_bound=max_distance)

    # Create mask for valid matches (finite distance = match found)
    match_mask = np.isfinite(distances)

    # Build matched catalog
    matched1 = catalog1[match_mask]
    matched2 = catalog2[indices[match_mask]]

    # Optionally combine columns from both catalogs
    matched_cat = hstack([matched1, matched2], table_names=[keys[0],keys[1]])
    return matched_cat


# ------------------------------------------------
# Background subtraction
# ------------------------------------------------
# TODO: need to include valid mask based on weight image or other metric

def _background_filename_for(image_path):
     """Return the standard cached background filename for an image."""
     base, _ = os.path.splitext(image_path)
     return f"{base}_background.fits"


def calculate_bkg(img,
          gal,
          band,
          box_size_pix=50,
          filter_size_pix=3,
          bkg_estimator=MedianBackground(),
          coverage_mask=False,
          doplot=True,
          sigma_to_clip_bkg=3.0,
          maxiters_for_bkg_clip=5,
          image_path=None,
          header=None,
          **kwargs):
     """Estimate the background model and save it to a standard background FITS file."""

     sigma_clip = SigmaClip(sigma=sigma_to_clip_bkg, maxiters=maxiters_for_bkg_clip)

     if type(box_size_pix) != type([]):
          box_size_pix = (box_size_pix, box_size_pix)
     if box_size_pix[0] % 2 == 0:
          box_size_pix = (box_size_pix[0] + 1, box_size_pix[1] + 1)

     if type(filter_size_pix) != type([]):
          filter_size_pix = (filter_size_pix, filter_size_pix)
     if filter_size_pix[0] % 2 == 0:
          filter_size_pix = (filter_size_pix[0] + 1, filter_size_pix[1] + 1)

     bkg_estimator = eval(bkg_estimator) if isinstance(bkg_estimator, str) else bkg_estimator

     if coverage_mask is False:
          print("Creating coverage mask")
          coverage_mask = (~np.isfinite(img)) | (img == 0)

     # note: photutils<3.0 used edge_method='pad' by default, but photutils>=3.0 does not have this option and instead pads with fill_value=0.0 by default.
     # Explicitly mask invalid and zero-valued pixels before the background estimate to avoid
     # the non-finite-data warning emitted by photutils.Background2D.
     bkg = Background2D(
          img,
          box_size=box_size_pix,
          filter_size=filter_size_pix,
          sigma_clip=sigma_clip,
          bkg_estimator=bkg_estimator,
          coverage_mask=coverage_mask,
     )

     rms_map = np.array(bkg.background_rms, dtype=float)
     valid_rms = (~coverage_mask) & np.isfinite(rms_map) & (rms_map > 0)
     # print(f"bkg array {bkg.background}")
     bkg_rms = np.nanmedian(rms_map[valid_rms]) if np.any(valid_rms) else np.nan
     bkg_mean = np.nanmean(np.asarray(bkg.background, dtype=float)[~coverage_mask])

     # print(f"bkg array {bkg.background}")
     if image_path is not None:
          background_path = _background_filename_for(image_path)
          out_header = header.copy() if header is not None else fits.Header()
          hdu = fits.PrimaryHDU(data=np.asarray(bkg.background, dtype=float), header=out_header)
          hdu.writeto(background_path, overwrite=True)
          print(f"Saved background map to {background_path}")

     # threshold_img = snr_threshold * bkg.background_rms
     img_sub = img - bkg.background

     if doplot:
          # Plot the image, background, and background-subtracted image
          fig, ax = plt.subplots(1, 3, figsize=(18, 6))
          norm = ImageNormalize(vmin=np.nanpercentile(img, 25.00),
                                vmax=np.nanpercentile(img, 99.99),
                                stretch=LogStretch())
          ax[0].imshow(img, origin='lower', cmap='inferno', norm=norm)
          ax[0].set_title(f"{gal.upper()} {band.upper()} mosaic")
          # TODO: gal and band as global properties
          ax[1].imshow(bkg.background, origin='lower', cmap='inferno')
          ax[1].set_title("Estimated background")
          img_sub = img - bkg.background
          norm_sub = ImageNormalize(vmin=np.nanpercentile(img_sub, 25.00),
                                    vmax=np.nanpercentile(img_sub, 99.99),
                                    stretch=LogStretch())
          ax[2].imshow(img_sub, origin='lower', cmap='inferno', norm=norm_sub)
          ax[2].set_title("Background-subtracted image")
          for a in ax:
               im = a.images[0]
               plt.colorbar(im, ax=a, pad=0.01, fraction=0.05)
          plt.savefig(out_dir + f"/{gal}_{band}_background_subtraction.png", dpi=300)
          plt.close(fig)

     print(f"Mean background: {bkg_mean}")
     print(f"Background rms: {bkg_rms}")
     return bkg.background, bkg_mean, bkg_rms


def subtract_bkg(image_path,
          gal=None,
          band=None,
          box_size_pix=50,
          filter_size_pix=3,
          bkg_estimator=MedianBackground(),
          coverage_mask=False,
          doplot=True,
          sigma_to_clip_bkg=3.0,
          maxiters_for_bkg_clip=5,
          **kwargs):
     """Create a cached background FITS file if needed and subtract it from the image."""
     background_path = _background_filename_for(image_path)

     if not os.path.exists(background_path):
          print(f"Background file not found for {image_path}. Calculating background...")
          img, err, snr_map, coverage_mask, header = open_jwst(image_path)
          calculate_bkg(
               img=img,
               gal=gal,
               band=band,
               box_size_pix=box_size_pix,
               filter_size_pix=filter_size_pix,
               bkg_estimator=bkg_estimator,
               coverage_mask=coverage_mask,
               doplot=doplot,
               sigma_to_clip_bkg=sigma_to_clip_bkg,
               maxiters_for_bkg_clip=maxiters_for_bkg_clip,
               image_path=image_path,
               header=header,
               **kwargs,
          )
     else:
          print(f"Background file found on disk for {image_path}.")

     img, err, snr_map, coverage_mask, header = open_jwst(image_path)
     bkg_background = fits.getdata(background_path)
     bkg_mean = np.nanmean(bkg_background)
     bkg_rms = np.nanstd(bkg_background)
     img_sub = img - bkg_background
     return img_sub, bkg_mean, bkg_rms, bkg_background


# ------------------------------------------------
# Source finding (using IRAF or findpeaks, DAO in progress)
# ------------------------------------------------
def run_source_finder(img, 
                      gal,
                      band,
          header, 
          bkg_rms,
          finder='iraf', 
          snr_threshold=3.0, 
          fwhm_pix=2.0, 
          box_size_pix=(5,5),  # TODO reconcile RH's value of 50 with JR's value of 3 here
          roundlo=-0.5, 
          roundhi=0.5, 
          sharplo=0.2, 
          sharphi=1.0, 
          nsources=10000,
          doplot=True,
          write=True, # write out finder catalog
          overwrite=True,
          **kwargs
     ):
     """Find sources in the image using IRAFStarFinder.
     Args:
          img: 2D array of background-subtracted image data
          header: FITS header of the image (used for WCS and pixel scale)
          finder: source finder to use (currently 'iraf' and 'peaks' supported)
          snr_threshold: signal-to-noise ratio threshold for source detection
          fwhm_pix: FWHM of the PSF in pixels (used for source detection)
          roundlo, roundhi: roundness limits for source selection
          sharplo, sharphi: sharpness limits for source selection
          nsources: if not None, only return this many brightest sources in the catalog
     Returns:
          sources: Table of detected sources with columns x_centroid, y_centroid, flux, sharpness, roundness, mag, peak, etc."""
     # Run the source finder
     print(f"Running source finder: {finder}")
     
     # Get the threshold image from the background calculation
     ths = snr_threshold * bkg_rms

     # Add option to import an external source catalog
     # instead of running a source finder. Useful for testing 
     # without regen and matching cats with HST/MUSE

     # IRAFStarFinder 
     if finder == 'iraf':
          source_finder = IRAFStarFinder(threshold=ths,
               fwhm=fwhm_pix,
               roundness_range=(roundlo, roundhi),
               sharpness_range=(sharplo, sharphi),
               n_brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)

     # DAOStarFinder (can use elliptical apertures)
     elif finder == 'dao':
          source_finder = DAOStarFinder(threshold=ths,
               fwhm=fwhm_pix,
               roundness_range=(roundlo, roundhi),
               sharpness_range=(sharplo, sharphi),
               n_brightest=nsources,
          )
          # Run the source finder
          sources = source_finder(img)

     elif finder == 'peaks':
          # find_peaks looks for local maxima above a specified threshold.
          # Requires a bit of extra work to get results in the same format as IRAFStarFinder/DAOStarFinder, 
          # and it doesn't calculate sharpness or roundness.
          # TODO: Add function converting find_peaks output to a table with xcentroid, ycentroid, flux, etc.

          # JR estimated a threshold from photutils with detect_threshold, which adds the background to a 
          # threshold map.  Instead, here we assume the background has already been subtracted and use a flat ths.

          # TODO: reconcile JR box_size_pix=3 with RH 50 

          # Check if box size is even. If it is, add one to each of the values
          if box_size_pix[0] % 2 == 0:
               box_size_pix = (box_size_pix[0] + 1, box_size_pix[1] + 1)
          sources = find_peaks(img, 
               threshold=ths, 
               box_size=box_size_pix,
               centroid_func=centroid_quadratic,
          )
          # For sources where the centroid could not be determined,
          # use the position of the peak instead.
          # RI: this does end up keeping sources on the edges which we don't want
          #----------------------------------------
          z=np.where(np.isnan(sources['x_centroid']))[0]
          if len(z)>0:
               print(f"Warning: discarding {len(z)} sources with NaN centroids")
          #      sources['x_centroid'][z]=sources['x_peak'][z]        
          #      sources['y_centroid'][z]=sources['y_peak'][z]
          z=np.where(np.isfinite(sources['x_centroid']))[0]
          sources = sources[z]


          # TODO not sure if the difference betwen x_centroid and xcentroid is used outside of this function
          sources['xcentroid']=sources['x_centroid']        
          sources['ycentroid']=sources['y_centroid']

     elif finder != 'iraf' and finder != 'dao' and finder != 'peaks':
          raise ValueError(f"Starfinder {finder} not recognized. Currently only 'iraf' and 'peaks' are supported.")

     #convert from x,y in the image to  sky coordinates   
     #----------------------------------------
     with warnings.catch_warnings():
          warnings.filterwarnings(
               "ignore",
               message=r".*OBSGEO.*",
               category=FITSFixedWarning,
          )
          wcs_obj = WCS(header)
     sk = wcs.utils.pixel_to_skycoord(sources['xcentroid'], sources['ycentroid'], wcs=wcs_obj)
     sources['ra']=sk.ra
     sources['dec']=sk.dec
     sources['ra_centroid']=sk.ra
     sources['dec_centroid']=sk.dec

     if doplot:
          # Plot the image with sources 
          fig, ax = plt.subplots(1, 1, figsize=(8, 8))
          norm = ImageNormalize(vmin=np.nanpercentile(img, 25.00), 
                                vmax=np.nanpercentile(img, 99.99), 
                                stretch=LogStretch())
          ax.imshow(img, origin='lower', cmap='inferno', norm=norm)
          ax.set_title(f"{gal.upper()} {band.upper()} mosaic")
          ax.scatter(sources['xcentroid'], sources['ycentroid'], s=10, edgecolor='cyan', facecolor='none', lw=0.5, alpha=0.2)
          im = ax.images[0]
          plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
          plt.savefig(out_dir+f"/{gal}_{band}_source_finder_{finder}.png", dpi=300)
          # zoom in on a region
          x0=np.mean(plt.xlim())
          y0=np.mean(plt.ylim())
          zoom_size = 100  # size of the zoomed-in region in pixels
          ax.set_xlim(x0 - zoom_size/2, x0 + zoom_size/2)
          ax.set_ylim(y0 - zoom_size/2, y0 + zoom_size/2)
          ax.set_title(f"{gal.upper()} {band.upper()} mosaic, S/N threshold = {snr_threshold}")
          ax.scatter(sources['xcentroid'], sources['ycentroid'], s=10, edgecolor='cyan', facecolor='none', lw=0.5, alpha=0.8)
          plt.savefig(out_dir+f"/{gal}_{band}_source_finder_{finder}_zoom.png", dpi=300)
          plt.close(fig)


     print(f"Found {len(sources)} sources")
     # print(sources.colnames)
     # put sources in decreasing order by peak flux:
     sources.sort('peak_value', reverse=True)

     if write:
          if "find_cat_filename" in kwargs:
               cat_name = kwargs["find_cat_filename"]
          else:
               cat_name = f"{gal}_jwst_{band}_find_cat." + cat_filetype
          print(f"Writing catalog to {out_dir + cat_name}")
          sources.write(out_dir + cat_name, overwrite=overwrite)

     return sources


# TODO: these things
def load_source_catalog():
     print("Load an external source catalog to use for photometry.")


def filter_catalog():
     print("Filtering catalog based on morphology and other criteria.")


# ------------------------------------------------
# Optimal aperture and photometry
# ------------------------------------------------
def get_optimal_aperture(data, sources, max_r=32, brightest=50, frac=0.95, doplot=True):
     """Find the optimal aperture radius to use for the photometry from the 
        curve of growth of the brightest n sources. 
     Args:
          data: 2D array of image data (background-subtracted)
          sources: Table of sources from source finder 
                   (must contain x_centroid, y_centroid, flux)
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

     print("Calculating optimal aperture...")
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

     if doplot:
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
          err,
          header, 
          sources, 
          gal, 
          band,
          radius=10, 
          radius_sky_in=12, 
          radius_sky_out=18, 
          use_brightest=False, 
          sigma_to_clip_bkg=3.0,
          maxiters_for_bkg_clip=5,
          phot_method='exact',
          doplot=True,
          write=True, 
          phot_cat_filename=None,
          overwrite=True,
          apcorr_method = None, 
          local_bkg_subtract=False,
          **kwargs):
     """Compute aperture photometry for sources and return catalog with RA, Dec, magnitudes, etc.
     
     Args:
          data: 2D array of image data (background-subtracted)
          err: 2D array of error data (same shape as data)
          header: FITS header of the image
          sources: Table of sources from source finder 
                   (must contain x_centroid, y_centroid, flux, sharpness, roundness, mag, peak_value)
          radius: radius of circular aperture to use for photometry (in pixels)
          radius_sky_in: inner radius of the sky annulus (in pixels)
          radius_sky_out: outer radius of the sky annulus (in pixels)
          use_brightest: if True, only use the brightest sources for photometry
          sigma_to_clip_bkg: sigma value for clipping background in the sky annulus
          maxiters_for_bkg_clip: maximum iterations for sigma clipping of background
          phot_method: method to use for photometry (e.g., 'exact', 'subpixel', etc.)
          doplot: if True, generate diagnostic plots
          write: if True, write catalog to out_dir with name {gal}_jwst_{band}_cat.fits
          phot_cat_filename: filename for the output photometry catalog
          overwrite: if True, overwrite existing catalog file
          apcorr_method: method for aperture correction (e.g., 'psf')
          local_bkg_subtract: if True, subtract local background from aperture photometry
          **kwargs: additional keyword arguments for photometry functions
          
     Returns:
          phot_full: Table with photometry results, including RA, Dec, aperture sum, magnitudes, etc.
     """

     if use_brightest is not False:
          # Aperture photometry of only brightest sources
          kbrightness = 'peak_value'
          if kbrightness not in sources.colnames:
               kbrightness = 'aperture_flux_mJy'
          sources = sources[np.argsort(sources[kbrightness])[::-1][:use_brightest]]
          print(f"using only {len(sources)} sources")

     if isinstance(radius_sky_in , str):
          if "r" in radius_sky_in:
               radius_sky_in = radius * float(radius_sky_in.split("r", 1)[0])
     if isinstance(radius_sky_out, str):
          if "r" in radius_sky_out:
               radius_sky_out = radius * float(radius_sky_out.split("r", 1)[0])
     print(f"Using sky annulus with inner radius {radius_sky_in} pixels and outer radius {radius_sky_out} pixels.")

     if apcorr_method == "psf":
          # Get PSF-based aperture correction parameters
          apcorr = get_apcorr_from_psf(band,radius,radius_sky_in,radius_sky_out)
          print(f"Using PSF-based aperture correction factor of {apcorr} for radius {radius} pixels.")
     elif apcorr_method != None:
          # Get aperture correction parameters from CRDS file
          radius, radius_sky_in, radius_sky_out, apcorr = get_apcorr_params(crds_dir, band, inst='NIRCam', **conf['parameters']['apcorr'])
          print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels.")

     # Do aperture photometry
     print()
     print(f"Doing aperture photometry for {len(sources)} sources...")
     kx = 'xcentroid'; ky = 'ycentroid'
     if kx not in sources.colnames or ky not in sources.colnames:
          kx = 'xcenter'; ky = 'ycenter'
     positions = np.transpose((sources[kx], sources[ky]))
     apertures = CircularAperture(positions, r=radius)
     aper_stats = ApertureStats(data, apertures, error=err)
     phot_full = aperture_photometry(data, apertures, error=err, method=phot_method) # Jimena also passed a mask - why?
     phot_full['aperture_min'] = np.asarray(aper_stats.min)

     # Annulus
     annuli = CircularAnnulus(positions, r_in=radius_sky_in, r_out=radius_sky_out)
     sigma_clip_bkg = SigmaClip(sigma=sigma_to_clip_bkg, maxiters=maxiters_for_bkg_clip)
     # mask = annuli.to_mask(method='exact')
     # Mask the data to exclude NaNs and infs from the background estimation
     mask = ((np.isinf(data)) | (np.isnan(data)))

     # Background annulus stats
     # TODO: Jimena did this in counts space, so there is potentially a sqrt(gain) factor that needs consideration. 
     bkg_stats = ApertureStats(data, annuli, sigma_clip=sigma_clip_bkg, mask=mask, sum_method=phot_method)
     bkg_median = bkg_stats.median
     bkg_median[np.isnan(bkg_median)]=0

     # Error on the flux due to background estimation uncertainty.
     # The worst case is that of structured background, where the error scales with the area of the aperture.  
     # The best case is that of unstructured background, where the error scales with the square root of the area of the aperture.  
     bkg_err_MJysrpix = bkg_stats.std * aper_stats.sum_aper_area.value
     bkg_err_mJy = bkg_err_MJysrpix * get_pixarea_in_sr(header) * 1e9
     bkg_err_scalefactor = np.sqrt(0.5*np.pi / bkg_stats.sum_aper_area.value)  # scale factor for background error based on area of annulus 

     # Subtract background from aperture sum
     phot_full['aperture_flux_mJy'] = phot_full['aperture_sum'] * get_pixarea_in_sr(header) * 1e9
     if local_bkg_subtract:
          phot_full['bkg_median_MJysr'] = bkg_median
          phot_full['bkg_flux_mJy'] = bkg_median * aper_stats.sum_aper_area.value * get_pixarea_in_sr(header) * 1e9
          phot_full['aperture_flux_mJy'] -= phot_full['bkg_flux_mJy']
     
     # Copy source-finder morphology columns
     if 'flux' in sources.colnames:  # it won't be there for findpeaks method.  TODO could be added in find step
          phot_full['finder_flux'] = np.asarray(sources['flux'])
     if 'sharpness' in sources.colnames:
          phot_full['sharpness'] = np.asarray(sources['sharpness'])
     if 'roundness' in sources.colnames:
          phot_full['roundness'] = np.asarray(sources['roundness'])          
     if 'mag' in sources.colnames:
          phot_full['finder_mag'] = np.asarray(sources['mag'])
     if 'peak' in sources.colnames:
          phot_full['peak'] = np.asarray(sources['peak'])
     elif 'peak_value' in sources.colnames: 
          phot_full['peak'] = np.asarray(sources['peak_value'])   # TODO change peakfinder output to have peak instead of peak_value

     # Include ra, dec
     with warnings.catch_warnings():
          warnings.filterwarnings(
               "ignore",
               message=r".*OBSGEO.*",
               category=FITSFixedWarning,
          )
          wcs = WCS(header)
     ra, dec = wcs.all_pix2world(phot_full["xcenter"], phot_full["ycenter"], 0)
     phot_full["ra"] = ra
     phot_full["dec"] = dec

     # Convert flux from the source finder in table (converted to AB magnitudes)
     if 'finder_flux' in phot_full.colnames:
          phot_full['finder_flux_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['finder_flux'], header=header)
     # Aperture sum from circular aperture photometry (converted to AB magnitudes)
     phot_full['aperture_sum_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header)
     # TODO add finder_flux_mjy

     if apcorr_method != None:
          if apcorr.unit.is_equivalent(u.dimensionless_unscaled):
               phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'] * apcorr, header=header)
          elif apcorr.unit.is_equivalent(u.mag):
               phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header) + apcorr.value
          # add aperture correction to aperture_flux_mJy 
          # TODO do we need to multiply the error by the apcorr? Jimena did not.
          if 'aperture_flux_mJy' in phot_full.colnames:
               phot_full['aperture_flux_mJy_apcorr'] = phot_full['aperture_flux_mJy'] * apcorr if apcorr.unit.is_equivalent(u.dimensionless_unscaled) else phot_full['aperture_flux_mJy'] + apcorr.value
     
     # TODO: Is there a better way to do this than a list?
     if band.lower()=='f335m' or band.lower()=='f770w' or band.lower()=='f1000w' or band.lower()=='f1130w' or band.lower()=='f2100w' or "pah" in band.lower():
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err_MJysrpix**2)
     elif band.lower()=='f200w' or band.lower()=='f300m' or band.lower()=='f360m' or band.lower()=='f444w':
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err_MJysrpix**2 * bkg_err_scalefactor**2)
     else:
          print(f"Band {band} not recognized for error calculation. Setting total_aperture_sum_err to max possible, sqrt(aperture_sum_err**2 + bkg_err**2).")
          phot_full['total_aperture_sum_err'] = np.sqrt(phot_full['aperture_sum_err']**2 + bkg_err_MJysrpix**2)

     # special step for the continuum subtracted PAH bands to filter some of the poor subtractions:
     if "pah" in band.lower():
          negative_threshold = -0.5
          z_neg = np.where(phot_full['aperture_min'] < negative_threshold)[0]
          phot_full['aperture_sum_err'][z_neg] *= 3
          if len(z_neg) > 0:
               print(f"Found {len(z_neg)} sources with aperture_min < {negative_threshold}. Increased their aperture_sum_err by a factor of 3.")


     # Add the errors
     phot_full['bkg_err_mJy'] = np.asarray(bkg_err_mJy)
     phot_full['poisson_err_mJy'] = np.asarray(phot_full['aperture_sum_err'] * get_pixarea_in_sr(header) * 1e9)
     phot_full['tot_err_mJy'] = np.asarray(phot_full['total_aperture_sum_err'] * get_pixarea_in_sr(header) * 1e9)
 
     # Print the column names of the photometry table
     # print(phot_full.colnames)

     # Write the catalog if requested
     if write:
          if phot_cat_filename is None:
               phot_cat_filename = f"{gal}_jwst_{band}_phot_cat_r{radius:4.2f}." + cat_filetype
          print(f"Writing catalog to {out_dir + phot_cat_filename}")
          phot_full.write(out_dir + phot_cat_filename, overwrite=overwrite)

     if doplot:
          # plot photometry results
          fig, ax = plt.subplots(1, 1, figsize=(6, 6))
          ax.plot(phot_full['aperture_flux_mJy'], phot_full['bkg_err_mJy']/phot_full['aperture_flux_mJy'], 'o', markersize=1, alpha=0.5, label="bg")
          ax.plot(phot_full['aperture_flux_mJy'], phot_full['poisson_err_mJy']/phot_full['aperture_flux_mJy'], 'o', markersize=1, alpha=0.5, label="poisson")
          ax.legend(loc="best",prop={"size":8})
          ax.set_xlabel("Flux (mJy)")
          ax.set_ylabel("Error/Flux")
          plt.xscale("log")
          plt.yscale("log")
          plt.savefig(out_dir+f"/{gal}_{band}_appflux_dflux.png", dpi=300)
          plt.close(fig)

          # Plot the image with significant sources 
          fig, ax = plt.subplots(1, 1, figsize=(8, 8))
          norm = ImageNormalize(vmin=np.nanpercentile(data, 25.00), 
                                vmax=np.nanpercentile(data, 99.99), 
                                stretch=LogStretch())
          ax.imshow(data, origin='lower', cmap='inferno', norm=norm)
          ax.set_title(f"{gal.upper()} {band.upper()}")
          # z=np.where(phot_full['aperture_flux_mJy']/phot_full['poisson_err_mJy']>5)[0]
          # ax.scatter(sources['xcentroid'][z], sources['ycentroid'][z], s=10, edgecolor='cyan', facecolor='none', lw=0.5, alpha=0.3,label="Poisson SNR>5")
          label_low_snr_values = False  # Set to True to label each red low-SNR source with its SNR value
          err_threshold = 1.5
          snr = phot_full['aperture_flux_mJy'] / phot_full['tot_err_mJy']
          hi_label_added = False
          lo_label_added = False
          for idx in range(len(phot_full)):
               x = phot_full['xcenter'][idx]
               y = phot_full['ycenter'][idx]
               linewidth = 1
               alpha = 0.7     
               if snr[idx] > err_threshold:
                    edgecolor = 'k'
                    label = None if hi_label_added else f"SNR > {err_threshold}"
                    hi_label_added = True
               elif snr[idx] >0 and phot_full['aperture_flux_mJy'][idx]/phot_full['poisson_err_mJy'][idx]>5:
                    edgecolor = 'g'
                    label = None if lo_label_added else f"SNR < {err_threshold}"
                    lo_label_added = True
               else:
                    continue
               circ = plt.Circle((x, y), radius=radius, fill=False, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, label=label)
               ax.add_patch(circ)
               if snr[idx] < err_threshold and label_low_snr_values:
                    #ax.text(x + radius * 1.2, y + radius * 1.2, f"{snr[idx]:.1f}", color='g', fontsize=8, ha='left', va='bottom')
                    ax.text(x + radius * 1.2, y + radius * 1.2, f"{phot_full['aperture_flux_mJy'][idx]/phot_full['poisson_err_mJy'][idx]:.1f}", color='g', fontsize=8, ha='left', va='bottom')
                    
          ax.legend(loc="best",prop={"size":8})
          im = ax.images[0]
          plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
          plt.savefig(out_dir+f"/{gal}_{band}_significant_sources.png", dpi=600)
          plt.close(fig)



          # Make the 6x6 plot of source cutouts using sources from the aperture corrected catalog
          # nonan=np.where(np.isfinite(phot_full['aperture_flux_mJy']))[0]
          # brightest_sources_apcorr = phot_full[nonan][np.argsort(phot_full['aperture_flux_mJy'][nonan])[::-1]][:36]
          # instead of re-sorting, leave these in the order from the finder which sorted them from brightest
          # peak to lower peak values
          brightest_sources_apcorr = phot_full[:36]
          # brightest_sources_apcorr = phot_full[nonan][:36]
          fig, axes = plt.subplots(6, 6, figsize=(10, 10), 
                                   gridspec_kw={'left':0.02,'bottom':0.02,'top':0.98,'right':0.98,
                                                'wspace': 0.01, 'hspace': 0.01})
          for i, (ax, row) in enumerate(zip(axes.flatten(), brightest_sources_apcorr)):
               x, y = row['xcenter'], row['ycenter']
               cutout_size = np.maximum(radius*4, radius_sky_out*1.1) # actually, half-cutout size
               x_min, x_max = int(x - cutout_size), int(x + cutout_size)
               y_min, y_max = int(y - cutout_size), int(y + cutout_size)
               cutout = data[y_min:y_max, x_min:x_max]
               norm_cutout = ImageNormalize(vmin=np.nanpercentile(cutout, 0), 
                                            vmax=np.nanpercentile(cutout, 100),
                                            # stretch=SqrtStretch()
               )
               ax.imshow(data, origin='lower', cmap='inferno', norm=norm_cutout)
               # Plot horizontal and vertical lines through the center of the cutout
               # ax.axhline(y, color='cyan', ls='--', lw=1.0)
               # ax.axvline(x, color='cyan', ls='--', lw=1.0)
               # Draw a circle with radius equal to the optimal aperture radius
               circle = plt.Circle((x, y), radius, edgecolor='red', facecolor='none', lw=1.5, alpha=1.0)
               sky_in_circle = plt.Circle((x, y), radius_sky_in, edgecolor='magenta', facecolor='none', lw=1.5, alpha=0.5)
               sky_out_circle = plt.Circle((x, y), radius_sky_out, edgecolor='magenta', facecolor='none', lw=1.5, alpha=0.5)  
               ax.add_patch(circle)
               ax.add_patch(sky_in_circle)
               ax.add_patch(sky_out_circle)
               ax.axis('off')
               # note: although display the entire image 36 times does keep the axes where they need to be
               # to overplot the other sources, it could lead to a large image, so may be better to 
               # render the cutout and subtract x_min, y_min from sources_in_cutout positions
               ax.set_xlim(x_min, x_max)
               ax.set_ylim(y_min, y_max)
               ax.text(0.5, 0.9, f"{row['aperture_flux_mJy']*1000:.1f}uJy ({row['bkg_flux_mJy']*1000:.2f})", color='white', fontsize=8, ha='center', va='center', transform=ax.transAxes)
               ax.text(0.08, 0.93, f"{i+1}", color='cyan', fontsize=8, ha='center', va='center', transform=ax.transAxes)
               # Select all sources in the catalog that are within the cutout region
               sources_in_cutout = phot_full[(phot_full['xcenter'] > x_min) & (phot_full['xcenter'] < x_max) & (phot_full['ycenter'] > y_min) & (phot_full['ycenter'] < y_max)]
               ax.scatter(sources_in_cutout['xcenter'], sources_in_cutout['ycenter'], s=50, edgecolor='cyan', facecolor='none', lw=1.0, alpha=0.5)
          plt.savefig(out_dir+f"/{gal}_{band}_cutouts_brightest_r{radius:4.2f}.png", dpi=400)
          plt.close(fig)




     return apertures, phot_full





#=====================================================================================
def bkg_error_quantiles(data, annulus_masks, sigma=3.0):
     """Compute local background statistics in the annulus around each source.
     Args:
          data: 2D array of image data (background-subtracted)
          annulus_masks: list of masks for each annulus (from annuli.to_mask())
          sigma_clip: SigmaClip object for sigma clipping the background pixels (optional)
          mask: boolean array where True indicates pixels to exclude from background estimation (e.g., low coverage or bad data)
     Returns:
     """
     
     bkg_10 = np.zeros(len(annulus_masks))
     bkg_90 = np.zeros(len(annulus_masks))
     bkg_10_clip = np.zeros(len(annulus_masks))
     bkg_90_clip = np.zeros(len(annulus_masks))
     npix_annulus = np.zeros(len(annulus_masks))
     npix_annulus_clipped = np.zeros(len(annulus_masks))
     sigma_clip = SigmaClip(sigma=sigma) if sigma is not None else None

     for i, m in enumerate(annulus_masks):
          annulus_data = m.multiply(data)

          if annulus_data is not None:
               # Flatten and remove zeros, NaNs, and infs
               annulus_data_1d = annulus_data[(annulus_data != 0) & np.isfinite(annulus_data) & ~np.isnan(annulus_data)] 

               if len(annulus_data_1d) > 0:
                    annulus_data_filtered = sigma_clip(annulus_data_1d) if sigma_clip is not None else annulus_data_1d
                    bkg_low, bkg_hi = np.quantile(annulus_data_1d, [0.1,0.9])
                    bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1,0.9])
                    # Update results
                    bkg_10[i] = bkg_low
                    bkg_90[i] = bkg_hi
                    bkg_10_clip[i] = bkg_low_clip
                    bkg_90_clip[i] = bkg_hi_clip
                    npix_annulus[i] = len(annulus_data_1d)
                    npix_annulus_clipped[i] = len(annulus_data_filtered)
          else:
               continue
     return bkg_10, bkg_90, bkg_10_clip, bkg_90_clip, npix_annulus, npix_annulus_clipped
                    




#=====================================================================================
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
               
               # TODO: get the pixel scale properly from header info
               # These correction factors are only valid for a specific radius.
               # pixel_scale = 0.031
               radius = 4 #* pixel_scale
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

def get_psf_file(band):
     candidates = glob.glob(psf_dir + f"PSF*{band}*fits")

     if len(candidates) == 0:
          raise FileNotFoundError(f"No PSF files found for {band} in {psf_dir}")
     elif len(candidates) > 1:
          print(f"Multiple PSF files found for {band} in {psf_dir}. Using the first one: {candidates[0]}")
     return candidates[0]     





#=====================================================================================
# Take an image clip and return the residual after subtracting the PSFcores of sources as defined by params
def residual(params,imclip,return_type,psfcore,scor_pix,sfitrgn_pix,njparams):
     # params = [fluxes,jtypes] # jtypes could be fixed
     # the "f" parameters are the fluxes of the main and neighboring sources
     fparms=[x for x in params.keys() if x[0]=='f']

     # TODO copy the imclip upstream, so this can just imclip without copying it every iteration

     nsrc=len(fparms) 
     if return_type == "substars":
          outclip = imclip.copy()
     elif return_type == "suball":
          outclip = imclip.copy() - params['bg']    
     elif return_type == "stars":
          outclip = np.zeros(imclip.shape) 
     else:
          raise ValueError(f"Unknown return_type: {return_type}")     
     mask=np.ones(outclip.shape,dtype="bool")
         
     for i in range(nsrc):
          dj,j=math.modf(params['j%i'%i])
          # x1,y1 parameter is the offset in full pixel units
          off=np.array([params['x%i'%i],params['y%i'%i]])
          off0=np.int32(off) # off0 is nearest full pixel  
  
          dx,x=math.modf(4*off[0]-4*off0[0]) # x,dx are in 1/4-pix units, and relative to the nearest whole pixel off0
          dy,y=math.modf(4*off[1]-4*off0[1])
          x=int(x)+2 # extra +2 because psf is centered between whole pixels
          y=int(y)+2
          j=int(j)
          
          # find broadened psfs bracketing the parameter j1 (and interpolate below)
          j1=j+1
          if j1>=njparams-1:
               j1=njparams-1

          # these psfs are also binned back to the original pixel scale from 4x
          q11 = np.roll(psfcore[j ],[x  ,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
          q21 = np.roll(psfcore[j ],[x+1,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
          q12 = np.roll(psfcore[j ],[x  ,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
          q22 = np.roll(psfcore[j ],[x+1,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
          j0model = q11*(1-dx)*(1-dy) +q12*(1-dx)*dy +q21*dx*(1-dy) +q22*dx*dy            
  
  
          # turn off qq calculation if njparams=1 i.e. no broadening
          if njparams>1:
               r11 = np.roll(psfcore[j1],[x  ,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
               r21 = np.roll(psfcore[j1],[x+1,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
               r12 = np.roll(psfcore[j1],[x  ,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
               r22 = np.roll(psfcore[j1],[x+1,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
               j1model = r11*(1-dx)*(1-dy) +r12*(1-dx)*dy +r21*dx*(1-dy) +r22*dx*dy
  
          # pp=np.roll(psfdat,[i,j],axis=[0,1])
          # gf(pp,widths[k]).reshape(s[0]//4,4,s[1]//4,4).sum(3).sum(1)
                                      
          if params['j%i'%i]>0:
               if j<njparams-1:
                    imodel=j0model*(1-dj) + j1model*dj
               else:
                    imodel=j1model
          else:
               imodel=j0model
  
          # now we need to full-pixel location of the imodel (psf core size scor_pix) inside the image clip
          brd=sfitrgn_pix//2-scor_pix//2 # border between clip and psfcore
          # where to place in x, if not off edge
          xou=[sfitrgn_pix[1]//2+off0[1]-scor_pix[1]//2, sfitrgn_pix[1]//2+off0[1]+scor_pix[1]//2]
          xin=[0,scor_pix[1]+1] # index in psfcore
          you=[sfitrgn_pix[0]//2+off0[0]-scor_pix[0]//2, sfitrgn_pix[0]//2+off0[0]+scor_pix[0]//2]
          yin=[0,scor_pix[0]+1] # index in psfcore
          if off0[1]<-brd[1]: # off left side
               xou[0]=0
               xin[0]=-off0[1]-brd[1]
          if off0[1]>=brd[1]: # off right side
               xou[1]=sfitrgn_pix[1]
               xin[1]=scor_pix[1]-(off0[1]-brd[1])
          if off0[0]<-brd[0]: # off bottom
               you[0]=0
               yin[0]=-off0[0]-brd[0]
          if off0[0]>=brd[0]: # off top
               you[1]=sfitrgn_pix[0]
               yin[1]=scor_pix[0]-(off0[0]-brd[0])
  
          if return_type=="stars": 
               outclip[xou[0]:xou[1],you[0]:you[1]] += \
                    params['f%i'%i]*imodel[xin[0]:xin[1],yin[0]:yin[1]]
          elif params['f%i'%i].vary==True:  # 2026 TODO make sure this is what we want - not subtract unfit neighbors
               outclip[xou[0]:xou[1],you[0]:you[1]] -= \
                    params['f%i'%i]*imodel[xin[0]:xin[1],yin[0]:yin[1]]
               mask[xou[0]:xou[1],you[0]:you[1]] = False
               # TODO mask tighter for psf peak only?
 
     if return_type=="suball":
          # TEST SOFTENING of residual - for f1000, SQRT helps, 1/3 doesn't make much difference.
          z=np.where(outclip>0)
          #outclip[z]=(outclip[z])**(1/4)
          outclip[z]=np.sqrt(outclip[z])
          z=np.where(outclip<0)
          # outclip[z]=-(-outclip[z])**(1/4)
          # outclip[z]=-np.sqrt(-outclip[z])
         
          outclip[np.where(mask)]=0
          return outclip # TODO - weight small scales by subtracting the median?
     else:
          return outclip





#=====================================================================================
def get_apcorr_from_psf(band,r_ap,r_sky_in,r_sky_out):
     """
     Get aperture correction from PSF for a given band and aperture parameters.

     Parameters
     ----------
     band : str
         The band of the observation.
     r_ap : array-like
         Aperture radii
     r_sky_in : array-like
         Inner radii of the sky annulus.
     r_sky_out : array-like
         Outer radii of the sky annulus.

     Returns
     -------
     apcor : array-like 
         Aperture correction factors for each set of aperture parameters.
     """
     apcor = np.zeros_like(r_ap, dtype=float)

     # Find the PSF file for this band and load the oversampled PSF image.
     psf_file = get_psf_file(band)
     psf_data = fits.getdata(psf_file)

     # Operate in oversampled (quarter-pixel) coordinates.
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

     for i in np.ndindex(r_ap.shape):
          aper = CircularAperture(pos, r=4.0 * r_ap[i])
          ann = CircularAnnulus(pos, r_in=4.0 * r_sky_in[i], r_out=4.0 * r_sky_out[i])
          ap_phot = aperture_photometry(psf_data, aper, method='exact')
          ann_phot = aperture_photometry(psf_data, ann, method='exact')

          npix_ap[i] = float(aper.area)
          npix_ann[i] = float(ann.area)
          psf_sum_ap[i] = float(ap_phot['aperture_sum'][0])
          psf_mean_ann[i] = float(ann_phot['aperture_sum'][0]) / npix_ann[i]

     return np.nansum(psf_data) / ( psf_sum_ap - psf_mean_ann * npix_ap ) * u.dimensionless_unscaled




#=====================================================================================
def fit_and_subtract(infile, # input mosaic image 
                    band="pah33",
                    whathdu='SCI', # which HDU to use for the image - TODO refactor into open_jwst
                    srcfile=None, # source catalog to use for fitting 
                    file_root="psffit",
                    pixbinfactor=1.,
                    fittype="amp", # "amp" or "amppos" or "ampwid" or None to just create residual
                    doplot=True,
                    kflux='flux_F1000W', # key name for app flux in input catalog
                    kdflux='fluxerr_F1000W',  # key name for app flux error in input catalog
                    kra='raj2000', # key name for RA in input catalog
                    kde='dej2000', # key name for Dec in input catalog
                    doregion=False,
                    rd0=[0,0], # center of region to fit in degrees
                    d=0.01, # size of region to fit in degrees
                    radius=None
                    ):

     from lmfit import minimize, Parameters, create_params


     maxfluxfactor=4 # limit on how much brighter the source can get when fit
    
     # rotational angle miri is V3 +~5
     # PA_APER =   245.97609259623073 / [deg] Position angle of aperture used
     # PA_V3   =   241.05116909076034 / [deg] Position angle of telescope V3 axis

     imfloor=1 # add a floor to the image for display only
     origfluxcut = 0.5 # filter for only sources with aperture flux above this value
     froot = file_root
     fiteverything=True # if False, only fit crowded sources
     plotborder=0 # fraction of sfitrgn_pix to include around the border when displaying the region
     alpha=0.5 # for overplotting sources
     larger_sfitrgn_pix=False
     tomjy=1.0 # convert from input catalog units to mJy TODO make this automatic based on the input catalog units
     debug=False # stop every source and show neighbors etc

     
     #---------------------------------------------
     # load input image
     with warnings.catch_warnings():
          warnings.filterwarnings(
               "ignore",
               message=r".*OBSGEO.*",
               category=FITSFixedWarning,
          )
          inhdu = fits.open(infile)[whathdu]
          inwcs = wcs.WCS(inhdu.header)

     # pixel scale in arcsec/pixel
     pixsize = wcs.utils.proj_plane_pixel_scales(inwcs) * 3600
     # if this is a simulated image moved to a larger distance, then the pixels are artificially large
     # because the simulated images have large pixels in the WCS, so that the coordinates of sources
     # match the original nearby image. If you're not doing simulated images, pixbinfactor should be 1.
     pixsize /= pixbinfactor
     
     # Sr per pixel, used to convert between mJy flux density and MJy/Sr peak value
     srperpix=(pixsize[0]/206265)**2

     fwhm_pix=filter_fwhm_pix[band.upper()]
     wave=filter_wave[band.upper()]


     # there are several radii and clip image sizes used here
     # first is rpix, the radius in pixels used for ds9 and plotting
     # position of first null = 2.37*HWHM
     rpix=2.37*fwhm_pix/2
     # radii related to psf-fitting are defined below.
     
     
     #---------------------------------------------
     # read in input source list
     srclist=Table.read(srcfile,format="ascii")
     
     # order from brightest to faintest
     # TODO refactor to do this only in the fit loop, to not disorder the actual list
     # and keep it identical to the input order
     u=np.argsort(srclist[kflux])[::-1]
     
     nsrc=len(srclist)
     srcra=srclist[kra].data
     srcde=srclist[kde].data
     
     
     #---------------------------------------------
     # set up psfs - expects 4 x oversampled psf in the file
     psffile = get_psf_file(band)

     psfdat=fits.getdata(psffile)
     #OVERSAMP=                    4 / Oversampling factor for FFTs in computation
     #DET_SAMP=                    4 / Oversampling factor for MFT to detector plane
     #PIXELSCL=               0.0277 / Scale in arcsec/pix (after oversampling)
     
     # make psfs broadened by "widths" quarter-pixel amounts:
     widths=np.int32(np.round(np.array([0,4,8])*wave/21))+1 # scale by wavelength
     # force broader - doesn't really do much for for miri, 
     widths=[1,3,6]     
     # 3.6sub has point sources at widths of 0.19" up to compact sources of 0.32"
     # 4 pixel width =16 quarter pix should get us up to the more extended sources
     nbroad=len(widths)
     
     s=psfdat.shape
     # this "p" array will store broadened psfs, at quarter-pix resolution
     p=np.zeros([nbroad,s[0],s[1]]) 
     
     # ROTATE psf to the correct position angle
     psfdat=rotate(psfdat,-inhdu.header['pa_aper'],reshape=False)
     
     for k in range(nbroad):
          p[k]=gf(psfdat,widths[k])
          # p[k]=gf(psfdat,widths[k]).reshape(s[0]//4,4,s[1]//4,4).sum(3).sum(1)
          # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
          p[k]/=p[k].sum()
     
     #---------------------------------------------
     # set up for psf-fitting 
     # the psf fitting will only use the core of the psf, defined here of size
     #   2*npixrcore in quarter-pixels = scor_qpix
     # the subimage over which the fitting is done is sized "sfitrgn_pix" in fullpixels,
     #    or "window" in degrees
     # sources inside rfit radius will be fitted
     # sources inside sfitrgn_pix, but outside of rfit, will be subtracted with their current flux in the flux catalog during the fit
     # then scor_pix is the size of the core psf in full pixels, scor_qpix//4
     
     # keep track of any sources that get their fluxes adjusted by fitting
     fitted=np.zeros(nsrc)
     # and the fitted j parameter (width)
     jout=np.zeros(nsrc)
     # and fitted xy position
     xyout=np.zeros([nsrc,2])
     # distance to the nearest source
     nearest=np.zeros(nsrc)
     # fitted local background/ floor level
     bgfit=np.zeros(nsrc)

     # the new fitted flux (will be filled during fitting)
     newflux=srclist[kflux].data.copy()
     
     # what is distance of proximity that sources have to be fit together?
     # use 2*fwhm
     rfit=fwhm_pix*pixsize[0]*2/3600
     
     # for F2100 I did 70:111 i.e. a half-width of 6.7*FWHM or 2.7*first null
     # npixrcore=int(round(7*fwhm/2/pixsize[0]*4)) # quarter-pixels
     # that seems a bit much 1000
     # npixrcore=int(round(6*fwhm_pix/4))*8 # quarter-pixels
     # but for 335-subtracted we seem to need even larger - there's a pix/bm issue here that we need to figure out
     npixrcore=int(round(3*fwhm_pix))*8 # quarter-pixels

     # scor_pix is npixrcore*2, modulo being even - this defines the size of the core psf used for fitting
     
     # this is the half-size of the input psf in quarter-pixels
     spsf=psfdat.shape[0]//2 # this is even - the psf is centered between pixels 
     icore=[spsf-npixrcore,spsf+npixrcore]

     # this is the core region of the psf in quarter-pixels
     psfcore=p[:,icore[0]:icore[1],icore[0]:icore[1]]
     # TODO: divide by sum of psfcore, to get the flux normalization right later when making the residual?
     scor_qpix=np.array(psfcore.shape[-2:])
     # scor_pix is the size of psfcore in full pixels
     scor_pix = scor_qpix//4  # has to be even
     
     # psfcore is centered at 39.5,39.5 = scor_pix/2-0.5
     degperpix=pixsize[0]/3600
     
     # sfitrgn_pix is the size of the fitting region in full pixels - it has to at least fit the core psf
     sfitrgn_pix=np.int32(np.round(scor_pix))*2
     # the distance in degrees within which we have to find sources to fit along with the source being considered
     window=(sfitrgn_pix[0]//2)*degperpix # equal to the half-size of the fitting region - find everything
     
     if larger_sfitrgn_pix:
          sfitrgn_pix *=2 
          froot+="_sfitrgnx2"
     
     if sfitrgn_pix[0]/2!=sfitrgn_pix[0]//2:
          raise ValueError("sfitrgn_pix is odd")
     
     
     if doregion:
          froot+="_region"
          cdec=np.cos(rd0[1]*np.pi/180)
          subim_xy=np.int32(inwcs.wcs_world2pix([[rd0[0]-d/cdec,rd0[1]-d],[rd0[0]+d/cdec,rd0[1]+d]],0))
          if larger_sfitrgn_pix: # does this work?  seems off?
               subim_xy+=np.array([[sfitrgn_pix[1]//4,-sfitrgn_pix[0]//4],[-sfitrgn_pix[1]//4,sfitrgn_pix[0]//4]])   
          subim=inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]
     else:
          subim=inhdu.data
          subim_xy=[[subim.shape[1],0],[0,subim.shape[0]]]

     # initialize the residual image for fitting, if there is going to actually be any fitting.
     if fittype is not None:
          resid=subim.copy()
          model=np.zeros(subim.shape)
     # params dx,dy are offsets from the central pixel of the stack for each src
     # with psfdat in hand, we go to closest quarter-pixels in the residual image
          
     # this will store the residual from photometry before fitting i.e. from the app phot
     resid_nofit=subim.copy()
     # this will just be the stars, using the original photometry
     model_nofit=np.zeros(subim.shape)
     
     
     
     #---------------------------------------------
     # subplot 0 is the original image, zoomed if doregion=True
     
     if doplot:          
          fig = plt.figure(0,figsize=(8,8))
          plt.clf()
          plt.subplots_adjust(top=0.95,bottom=0.1,left=0.1,right=0.98)
          plt.subplot(2,2,1)
          norm = ImageNormalize(vmin=np.nanpercentile(subim, 10),
                                vmax=np.nanpercentile(subim, 99.99),
                                stretch=LogStretch())
          cmap4ims = "viridis"
          plt.imshow(subim, origin='lower', cmap=cmap4ims, norm=norm)
          if doregion:
               ax=plt.gca()
               #ax.use_sticky_edges=False
               #ax.margins(x=-0.4,y=-0.4)
               # zoom because of border
               s=subim.shape
               ax.set_xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
               ax.set_ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
          
          plt.xlabel("k:fit m:near fit x:near nofit r:nofit")
          plt.title("original image")
          plt.xticks([])
               
     # open a ds9 output file
     ds9reg=open(f"{srcfile[:-4]}.reg","w")
     ds9reg.write("fk5\n")
     
     th=np.arange(21)/10*np.pi
     st=np.sin(th)
     ct=np.cos(th)
     
     
     #---------------------------------------------
     # the loop over sources
     for i0 in range(nsrc):

          i=u[i0] # sorted order bright->faint
          if (srclist[kflux][i]<1e-8):
               continue
          # discard sources with flux below the threshold
          if srclist[kflux][i]<(origfluxcut*srclist[kdflux][i]):
               newflux[i]=1e-8
               fitted[i]=-1
               continue

          print(f"Processing source {i0} (sorted index {i})")
          # pixel position of the source in the subimage
          xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
          
          # only deal with sources where the entire regular-pix-size psfcore (sfitrgn_pix) fits within the subim
          # (the subim is the entire image if doregion=False)
          if srclist[kflux][i]>0 and xy.min()>(sfitrgn_pix[0]/2) and xy[1]<(subim.shape[0]-sfitrgn_pix[0]//2) and xy[0]<(subim.shape[1]-sfitrgn_pix[1]//2):
      
               if debug:
                    print(i,xy)
       
               xy0=np.int32(np.round(xy))
               # from here to the end of processing this source, everything is relative to xy0,
               # the rounded source position in the original (sub)image coordinates, 
               # which maps to sfitrgn_pix//2 in the small clipped region to be fit
       
               # fractional pixel offset of the source
               off=xy-xy0
               
               # determine nearest 1/4 pixel offset to roll the psf        
               # +2 is b/c psf is centered between whole pixels - verified 20250209
               xy4i=np.int32(np.round(off*4))+2
               # subtract the original aperture photom, unbroadened
               # TODO consider subtracting entire psf, not just core of size scor_pix
               resid_nofit[xy0[1]-scor_pix[1]//2:xy0[1]+scor_pix[1]//2,xy0[0]-scor_pix[0]//2:xy0[0]+scor_pix[0]//2]-= \
                    srclist[kflux][i]*tomjy*1e-9/srperpix* \
                    np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
               model_nofit[xy0[1]-scor_pix[1]//2:xy0[1]+scor_pix[1]//2,xy0[0]-scor_pix[0]//2:xy0[0]+scor_pix[0]//2]+= \
                    srclist[kflux][i]*tomjy*1e-9/srperpix* \
                    np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)

               # if this source has not yet been fitted
               # (if fittype==None, then we'll never go in this block)
               if fittype and fitted[i]<=0:
                   # find neighboring sources, within "window" (usually set to sfitrgn_pix*pixsize/2 above)
                    cdec=np.cos(srcde[i]*np.pi/180)
                    znear=np.where( (np.absolute(srcra-srcra[i])<1.*window/cdec )*
                                   (np.absolute(srcde-srcde[i])<1.*window )*
                                   (newflux>1e-6))[0]
       
                    # if newflux (which is still the input flux) of the main source is bright enough, 
                    # and there are some neighbours, they may be part of the airy ring and should be removed
                    if newflux[i]>0.15 and len(znear)>1:    # TODO make this condition specific to F1000W only
                         d2=(srcra[znear]-srcra[i])**2*cdec**2 + (srcde[znear]-srcde[i])**2
                         zring=znear[np.where((d2>(1.1/3600)**2)*(d2<(1.5/3600)**2))[0]] # TODO THIS IS FOR F1000W ONLY
                         if len(zring)>0:
                              fitted[zring]=100
                              newflux[zring]=1e-8
                              # TODO plot these with different symbol?
                         znear=np.where( (np.absolute(srcra-srcra[i])<1.*window/cdec )*
                                       (np.absolute(srcde-srcde[i])<1.*window )*
                                       (newflux>1e-8))[0]
       
                    if len(znear)>1: # sort neighbour sources by distance from the primary source
                         d2=(srcra[znear]-srcra[i])**2*cdec**2 + (srcde[znear]-srcde[i])**2
                         znear_ord=znear[d2.argsort()]
                         d2.sort()

                         # distinguish between "very" close - within 1/4 of the fitting radius, and just neighboring 
                         if d2[1] < (rfit**2 / 16):  # /4 or /9= TEST ALLOW CLOSER TO BE WIDENED
                              someclose=True  # some are very close
                         else:
                              someclose=False
                    else: # znear includes the source itself, so if it is length==1, its *only* the primary source
                         znear_ord=znear
                         someclose=False
                         d2=np.array([0])

                    # fiteverything means to fit all neighbors in the fitregion, 
                    # otherwise only fit the primary source and the "very close"
                    if fiteverything or someclose:
                         if "wid" in fittype and not someclose: # extra parameters - only do widening if not crowded.
                              # print("widening")
                              njparams=nbroad
                         else:
                              njparams=1
       
                         # NOTE: sfitrgn_pix must be even #pix (20250210 TODO check if still true)
                         # finally, we actually extract the "clip" fitting region:
                         imclip=resid[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                                      xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2]

                         # the fitting will deal with the source peak flux in MJy/sr, 
                         # because its easier to scale the model PSF that way, and later we can 
                         # convert that back to mJy  
                         thisfluxMJy=newflux[i]*tomjy*1e-9/srperpix # mJy -> Mjy/sr assuming normalized psf
         
                         params = Parameters()
                         params.add('bg', value=np.mean(imclip), min=0)

                         # allow the fitted flux to be between 0 and the maximum flux factor times the initial flux
                         params.add('f0', value=thisfluxMJy,min=0,max=thisfluxMJy*maxfluxfactor)
         
                         # if we're allowing the PSF to widen, we let the "j0" parameter vary - 
                         # j0 is the index in the list of PSF broadening widths, and array of psfcores prepared above
                         if njparams<=1:
                              params.add('j0',value=0,vary=False)
                         else:            
                              params.add('j0',value=0,min=0,max=njparams-1)
         
                         # this will be the fitted offset from xy0 in whole pixel units, 
                         # and is allowed to vary by +/- 1 pixel in both x and y directions
                         params.add('x0',value=off[0],min=off[0]-1,max=off[0]+1)  
                         params.add('y0',value=off[1],min=off[1]-1,max=off[1]+1)

                         # if the position is not being fitted, we fix those offset values x0 and y0
                         if "pos" not in fittype:
                              params['x0'].vary=False
                              params['y0'].vary=False

                         # now go through the neighbors
                         for ii in np.arange(1,len(znear_ord)):
                             
                              # other sources positions, are offset relative to the main source
                              xy_near=inwcs.wcs_world2pix([[srcra[znear_ord[ii]],srcde[znear_ord[ii]]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
                              off=xy_near-xy0

                              # f1 f2 etc are the (peak) fluxes of the neighboring sources in MJy/sr
                              thisfluxMJy=newflux[znear_ord[ii]]*tomjy*1e-9/srperpix # original or fitted flux
                              params.add('f%i'%ii, value=thisfluxMJy,min=0,max=thisfluxMJy*maxfluxfactor)
         
                              # don't re-fit previously fitted sources, nor ones outside the rfit fitting radius
                              if d2[ii]>rfit**2 or fitted[znear_ord[ii]]>0:
                                   params['f%i'%ii].vary=False
         
                              if njparams<=1:
                                   params.add('j%i'%ii,value=0,vary=False)
                              else:
                                   params.add('j%i'%ii,value=0,min=0,max=njparams-1)
                             
                              params.add('x%i'%ii,value=off[0],min=off[0]-1,max=off[0]+1)
                              params.add('y%i'%ii,value=off[1],min=off[1]-1,max=off[1]+1)
          
                              if "pos" not in fittype:
                                   params['x%i'%ii].vary=False
                                   params['y%i'%ii].vary=False
         
                         # TODO put upper bound on bg level
         
                         # lmfit can't deal with nans
                         z=np.where(np.isnan(imclip))
                         # srcstack[0][z]=0
                         # but the nans are on the edge so just don't fit sources with NaNs nearby
                         if len(z[0])>0:
         
                              # allow to get within scor_pix of NaNs (generally only off the edge)
                              s=imclip.shape
                              zctr=np.where(np.isnan(imclip[s[1]//2-scor_pix[1]//2:s[1]//2+scor_pix[1]//2,s[0]//2-scor_pix[0]//2:s[0]//2+scor_pix[0]//2]))[0]
                              if len(zctr)>0:
                                   print("NaN too near position")
                                   continue
                              else:
                                   # if there is a stray NaN in the fitting region, set it to zero
                                   imclip[z]=0
         
                         # https://lmfit.github.io/lmfit-py/
                         out=minimize(residual,params, args=[imclip,"suball",psfcore,scor_pix,sfitrgn_pix,njparams],nan_policy="omit")

                         if out.success:
                              # TODO this may be ~10% high because we fit the PSF core vs full PSF
                              newflux[i] = out.params['f0'].value/tomjy *srperpix/1e-9
                              fitted[i]=len(np.where(d2<rfit**2)[0]) # number of sources fitted TODO this doesn't take into account if some near sources have not be re-fit because they were previously fit
                              jout[i]=out.params['j0']
                              xyout[i]=np.array([out.params['x0'].value,out.params['y0'].value])+xy0
                              bgfit[i]=out.params['bg']
          
                              # plot the point in black if it has been fitted
                              if debug:
                                   plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'k',alpha=alpha,linewidth=1)
                              # set nearby *fitted* values and jout, xyout,
                              # so they don't get fit again later
                              for ii in np.arange(1,len(znear_ord)):
                                   xynear=np.array([out.params['x%i'%ii].value,out.params['y%i'%ii].value])+xy0
                                   xyout[znear_ord[ii]]=xynear
                                   if out.params['f%i'%ii].vary==True:
                                        fitted[znear_ord[ii]]=fitted[i]
                                        jout[znear_ord[ii]]=out.params['j%i'%ii]
                                        bgfit[znear_ord[ii]]=out.params['bg']
                                        # plot the neighbor as magenta if it has been fitted
                                        if debug:
                                             plt.plot(xynear[0]+ct*rpix,xynear[1]+st*rpix,'m',alpha=alpha,linewidth=2)

                                        # subtract the fitted neighbor from the original-photometry residual image
                                        # we do this here because we won't consider a neighbor that gets fitted again later
                                        # if we're doing actual fitting.
                                        # If we're not doing actual fitting we'll never get here, 
                                        # so this subtraction only happens during actual fitting.
                                        xy0near=np.int32(np.round(xynear))                                        
                                        xy4i=np.int32(np.round((xynear-xy0near)*4))
                                        resid_nofit[xy0near[1]-scor_pix[1]//2:xy0near[1]+scor_pix[1]//2,xy0near[0]-scor_pix[0]//2:xy0near[0]+scor_pix[0]//2]-= \
                                             srclist[kflux].data[znear_ord[ii]]*tomjy*1e-9/srperpix* \
                                             np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
                                        
                                        model_nofit[xy0near[1]-scor_pix[1]//2:xy0near[1]+scor_pix[1]//2,xy0near[0]-scor_pix[0]//2:xy0near[0]+scor_pix[0]//2]+= \
                                             srclist[kflux].data[znear_ord[ii]]*tomjy*1e-9/srperpix* \
                                             np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
          
                                      
                                   # if we didn't fit this neighbor, and it wasn't previously fitted, plot it as a black 'x'
                                   elif fitted[znear_ord[ii]]==0 and debug:
                                        #fitted[znear_ord[ii]]=-1
                                        plt.plot(xynear[0],xynear[1],'kx',alpha=alpha,linewidth=1)
                              # if this source has been successfully fit,
                              # replace its params with the new fitted ones 
                              params=out.params 
         
                         elif debug:  # failed fit
                              fitted[i]=-1
                              pdb.set_trace()
                             
                         if len(znear)>1:
                              nearest[i]=np.sqrt(d2[1])
       
                    # calling residual(return_type=substars) will only remove the fitted sources and 
                    # not remove a flat bg i.e. its designed for creating the residual image
                    thisresid=residual(params,imclip,"substars",psfcore,scor_pix,sfitrgn_pix,njparams)
                    # note, failed fits still get subtracted from the fitted resid
                    # TODO subtract entire psf, not just core, here (would need the residual function to do all that functionality now, so probably not worth it
                    thismodel=residual(params,imclip,"stars",psfcore,scor_pix,sfitrgn_pix,njparams)
       
                    resid[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                         xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2] = thisresid

                    model[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                         xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2] = thismodel

               # / end of "if actually fitting this source" block
       
               # if this source is not fit because it was previously fit, or
               # we're only fitting crowded sources, then plot it as a dashed red circle
               elif debug:
                    plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'r--',dashes=(2,7),linewidth=1)
       
               if debug:
                    plt.ion()
                    plt.show()
                    pdb.set_trace()
      
          else: # this source is outside of the fittable region of the (sub)image
               fitted[i]=-2
      
                  
          # write all sources to ds9 TODO change symbol by fit outcome
          if newflux[i]/srclist[kdflux][i]>3:
               ds9reg.write("circle(%f,%f,1e-4)\n"%(srcra[i],srcde[i]))
          #else:
          #    ds9reg.write("point(%f,%f) # point=x color=white\n"%(srcra[i],srcde[i]))
             
     ds9reg.close()
     
     
     
     # =============== display the original/apphot residual image
     if doplot:
          plt.subplot(2,2,2)
          plt.imshow(resid_nofit,norm=norm,cmap=cmap4ims,origin="lower")
          if doregion:
              s=subim.shape
              plt.xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
              plt.ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
              
          if debug:
              plt.xlabel("k:fit r:nofit y:fitfail")
          plt.title("apphot residual")
          plt.xticks([])
      
          if debug:
              for i in range(nsrc):
                  xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
                  if newflux[i]>-10:
                      if fitted[i]!=0:
                          plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'k',alpha=alpha,linewidth=1)
                      else:
                          plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'r:',alpha=alpha,linewidth=1)
                  else:
                      plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'y',linewidth=3)
     
     

     if fittype is not None:
          if doplot:
               # ========================================
               # now show the residual after psf-fitting 
               plt.subplot(2,2,3)
               plt.imshow(resid, origin='lower', cmap=cmap4ims, norm=norm)
               if debug:
                    plt.xlabel("k:fit r:v.fnt m:~fnt c:~brt, y:fail")
               plt.title("fit residual: "+fittype)
               plt.xticks([])
               
               if debug:
                    for i in range(nsrc):
                         xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
                         if fitted[i]!=0:
                              if newflux[i]>-10:
                                   if newflux[i]<1e-4 and srclist[kflux][i]>1e-3:
                                        plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'r:',linewidth=1)
                                   elif newflux[i]<0.5*srclist[kflux][i] and srclist[kflux][i]>1e-3 and newflux[i]>1e-4:
                                        plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'m',linewidth=1)
                            
                                   elif newflux[i]>2*srclist[kflux][i] and srclist[kflux][i]>1e-4 and newflux[i]>1e-4:
                                        plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'c',linewidth=1)
                            
                                   else:
                                        plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'k',alpha=alpha,linewidth=1)
                              else:
                                   plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,'y',linewidth=3)

          # save the residual image after fitting
          if doregion:
               s=subim.shape
               plt.xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
               plt.ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
               inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=resid
               inhdu.writeto(infile[:-5]+"_resid_region_"+fittype+".fits",overwrite=True)
          else:
               inhdu.data=resid
               inhdu.writeto(infile[:-5]+"_resid_"+fittype+".fits",overwrite=True)

          # save the model image after fitting
          if doregion:
               inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=model
               inhdu.writeto(infile[:-5]+"_model_region_"+fittype+".fits",overwrite=True)
          else:
               inhdu.data=model
               inhdu.writeto(infile[:-5]+"_model_"+fittype+".fits",overwrite=True)


     # save the residual image without fitting
     resid_suffix = f"_resid_apphot_r{radius:4.2f}" if radius is not None else "_resid_apphot"
     if doregion:
          inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=resid_nofit
          inhdu.writeto(infile[:-5]+resid_suffix+"_region.fits",overwrite=True)
     else:
          inhdu.data=resid_nofit
          inhdu.writeto(infile[:-5]+resid_suffix+".fits",overwrite=True)
     # save the model image without fitting
     model_suffix = f"_model_apphot_r{radius:4.2f}" if radius is not None else "_model_apphot"
     if doregion:
          inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=model_nofit
          inhdu.writeto(infile[:-5]+model_suffix+"_region.fits",overwrite=True)
     else:
          inhdu.data=model_nofit
          inhdu.writeto(infile[:-5]+model_suffix+".fits",overwrite=True)


     if fittype is not None:
          srclist['ra'] = srcra
          srclist['dec'] = srcde
          srclist.add_columns([newflux,jout,fitted],names=[kflux+"_refit_"+fittype,('jout_%4.1f_'%wave)+fittype,('nfitted_%4.1f_'%wave)+fittype])
          srclist.write(froot+"_refit.csv",overwrite=True)
     
     
     if doplot:
          # standard overplotting (non-debug)
          for i in range(nsrc):
               xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
               if newflux[i]>1e-8 and xy.min()>0 and xy[1]<subim.shape[0] and xy[0]<subim.shape[1] and srclist[kflux][i]>1e-6:
                   if fitted[i]>0:
                       if srclist[kflux][i]<=0:
                           col='r'
                           wid='2'
                           alpha=1
                       else:
                           fratio=newflux[i]/srclist[kflux][i]
                           alpha=1
                           wid=1
                           if fratio<0.1:
                               col='r'
                               alpha=0.5
                           elif fratio<0.5:
                               col='m'
                           elif fratio>3:
                               col='c'
                           else:
                               col='k'
                               alpha=0.5
           
                       plt.subplot(2,2,4)
                       plt.plot(srclist[kflux][i],newflux[i],'.',color=col)
           
                       if not debug:
                           for jj in [1,2,3]:
                               plt.subplot(2,2,jj)
                               plt.plot(xy[0]+ct*rpix,xy[1]+st*rpix,col,alpha=alpha,linewidth=wid)
                   elif not debug: # not fitted points
                       for jj in [1,2,3]:
                           plt.subplot(2,2,jj)
                           plt.plot(xy[0],xy[1],'ks',alpha=0.5,markersize=5,linewidth=1,mfc="none")
     
     
     
          plt.subplot(2,2,4)
          plt.plot(plt.xlim(),plt.xlim(),'k',alpha=0.2)
          plt.xscale("log")
          plt.yscale("log")
          plt.xlabel("aperture photometry")
          plt.ylabel("psf-fitted photometry")
     
          plt.savefig(froot+"_fit_residuals.png")
          plt.close()












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



















#=====================================================================================
# MAIN 

# Directories
jwst_dir = local['jwst_dir']
out_dir = local['out_dir']
psf_dir = local['psf_dir']
crds_dir = local['crds_dir']
cat_path = local['out_dir']  
# TODO: add cat_path to local.toml if we want to load in an external catalog for photometry instead of running a source finder.

print("-----------------------------------------")
print(f"JWST data directory: {jwst_dir}")
print(f"Output directory: {out_dir}")
print("-----------------------------------------")

# Check that input data directory exists
if not os.path.exists(jwst_dir):
     raise FileNotFoundError(f"JWST data directory {jwst_dir} does not exist. Please check the path in the config file.")
     exit()

# Check that out_dir exists
if not os.path.exists(out_dir):
     raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
     exit()


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
          steps: list of steps to run (e.g., ['bkg_subtract', 'subtract_bkg', 'source_find', 'r_opt', 'aperture_photometry'])
          targets: list of galaxy names to process
          use_filter_fwhm: this will eventually go into the config
          conf: dictionary of parameters from the config file."""

     print(" ")
     catalogs = {}

     for gal in targets:
          # loop through the filters for this galaxy
          for band in bands:
               print("-----------------------------------------")
               print(f">>> Processing {gal} at {band}...")

               # Get the full path to the data
               datafile = get_file(
                    wdir=jwst_dir, 
                    version=version, 
                    project=projects[0], 
                    galaxy=gal,
                    ptype=ptype[0],
                    filter=band)

               # Initialise catalogs dict to store the photometry results for each galaxy and filter
               if gal not in catalogs:
                    catalogs[gal] = {}
               if band not in catalogs[gal]:
                    catalogs[gal][band] = {}

               # Open the JWST data file 
               img, err, snr_map, coverage_mask, header = open_jwst(datafile)
               # TODO get distance from the galaxy sample table intead of the config file
               # Subtract background 
               if 'subtract_bkg' in steps:
                    print()
                    print(f"Subtracting background for {gal} at {band}...")
                    if 'box_size_pix' not in conf['parameters']['bkg_subtract']:
                         # Convert box size from pc to pixels using the pixel scale from the header
                         pix_scale = get_pixarea_in_sr(header) ** 0.5 * (180/np.pi) * 3600  # arcsec/pixel
                         box_size_pc = conf['parameters']['bkg_subtract']['box_size_pc']
                         box_size_pix = int(box_size_pc * 206265 / (pix_scale * conf['parameters']['bkg_subtract']['dist_Mpc'] * 1e6 ))
                         conf['parameters']['bkg_subtract']['box_size_pix'] = box_size_pix

                    if 'filter_size_pix' not in conf['parameters']['bkg_subtract']:
                         # Convert filter size from pc to pixels using the pixel scale from the header
                         pix_scale = get_pixarea_in_sr(header) ** 0.5 * (180/np.pi) * 3600  # arcsec/pixel
                         filter_size_pc = conf['parameters']['bkg_subtract']['filter_size_pc']
                         filter_size_pix = int(filter_size_pc * 206265/ (pix_scale * conf['parameters']['bkg_subtract']['dist_Mpc'] * 1e6 ))
                         conf['parameters']['bkg_subtract']['filter_size_pix'] = filter_size_pix

                    img_sub, bkg_mean, bkg_rms, bkg_background = subtract_bkg(
                         image_path=datafile,
                         gal=gal,
                         band=band,
                         **conf['parameters']['bkg_subtract'],
                    )
                    use_image = img_sub
               else:
                    # bkg_rms is passed to the source finder even if background subtraction is not performed
                    # so we need a different rms estimate;  TODO: implement a proper background RMS estimation for non-subtracted images
                    # For now, we use the standard deviation of the image as a rough estimate of the background RMS
                    bkg_rms = np.nanstd(img)
                    use_image = img
               
               # Load the filename from the config
               if "find_cat_filename" in conf['parameters']['source_find']:
                    cat_filename = conf['parameters']['source_find']['find_cat_filename']
               else:
                    cat_filename = f"{gal}_jwst_{band}_find_cat." + cat_filetype

               if 'source_find' in steps:
                    # Get sources using the source finder
                    print()
                    print(f"Finding sources for {gal} at {band}...")
                    sources = run_source_finder(
                         img=use_image, 
                         gal=gal,
                         band=band,
                         header=header, 
                         bkg_rms=bkg_rms, 
                         **conf['parameters']['source_find'],
                    )
               
               else:
                    print()
                    print(f"Importing sources from existing catalog for {gal} at {band}...")
                    # Load the existing source catalog
                    sources = Table.read(cat_path + cat_filename)
                    print(f"Loaded {len(sources)} sources from {cat_path + cat_filename}")

                    # Check if any of the filename contains the word 'dolphot'
                    # If so, then we need to do something a bit different with the colnames. 
                    # RI: I would prefer to fix the dolphot catalog upstream and have the right columns by the time we get here, 
                    # but that might be more work than just doing this here.
                    if 'dolphot' in cat_filename.lower():
                         print ("Using the dolphot catalog.")
                         current_wcs = WCS(header)
                         xcentroid, ycentroid = current_wcs.all_world2pix(sources['RA_deg'], sources['Dec_deg'], 0)
                         sources['xcentroid'] = xcentroid
                         sources['ycentroid'] = ycentroid
                    elif 'ra' in sources.colnames and 'dec' in sources.colnames:
                         with warnings.catch_warnings():
                              warnings.filterwarnings(
                                   "ignore",
                                   message=r".*OBSGEO.*",
                                   category=FITSFixedWarning,
                              )
                              current_wcs = WCS(header)
                         xcentroid, ycentroid = current_wcs.all_world2pix(
                              sources['ra'], sources['dec'], 0
                         )
                         sources['xcentroid'] = xcentroid
                         sources['ycentroid'] = ycentroid

                    # Checks that the colnames include x_centroid, y_centroid, flux, sharpness, roundness, mag, peak, etc. 
                    # and print a warning if any are missing
                    required_cols = ['xcentroid', 'ycentroid', 'flux']
                    x_to_search_for = ['xcentroid', 'x_center', 'x_centroid', 'xcenter']
                    y_to_search_for = ['ycentroid', 'y_center', 'y_centroid', 'ycenter']
                    # Cycle through
                    for col in required_cols:
                         if col not in sources.colnames:
                              # Check whether there is a xcenter and ycenter column instead of x_centroid and y_centroid, and if so, rename them
                              if col == 'xcentroid' and 'x_center' in sources.colnames:
                                   sources['xcentroid'] = sources['x_center']
                              elif col == 'ycentroid' and 'y_center' in sources.colnames:
                                   sources['ycentroid'] = sources['y_center']
                              else:
                                   print(f"Warning: Column '{col}' is missing from the external catalog."
                                        f"\n Please make sure the catalog has the required columns: {required_cols}.")

               # **** Alternatively, load in a catalog computed by another method here ****
               # TODO: if loading in another catalog, need a path to it in local.toml. 

               # Either get the optimum radius based on curve of growth...
               if 'r_opt' in steps:
                    print(f"Computing optimal aperture for photometry...")
                    if 'subtract_bkg' not in steps:
                         print("Warning: r_opt step is being run without background subtraction. This may affect the results.")
                    r_opt = get_optimal_aperture(
                         data = use_image,
                         sources = sources,
                         **conf['parameters']['r_opt']
                    )
               else:
                    fwhm2rad = conf['parameters']['photometry'].get('fwhm2rad', 2.5)
                    r_opt = filter_fwhm_pix[band.upper()] * fwhm2rad if use_filter_fwhm else conf['parameters']['photometry']['aperture_radius']
                    if 'aperture_photometry' in steps:
                         print(f"Using fixed aperture radius of {r_opt} pixels for photometry.")

               # Update the fwhm according to the filter if use_filter_fwhm is True.
               # If use_filter_fwhm is False, stay at specified value.
               if use_filter_fwhm and 'aperture_photometry' in steps:
                    try:
                         fwhm_pix = filter_fwhm_pix[band.upper()]
                         print(f"Using FWHM of {fwhm_pix} pixels for source detection based on JWST PSF for {band.upper()}.")
                    except KeyError:
                         print(f"Warning: FWHM for {band.upper()} not found in filter_fwhm_pix dictionary. Using default FWHM of {fwhm_pix} pixels for source detection.")

               # TODO: is there a better way of doing this?
               if "apcorr" in steps:
                    apcorr_method = conf['parameters']['apcorr']['apcorr_method']
               else:
                    apcorr_method = None

               # # ...or just set it to a fixed value (e.g., based on the PSF FWHM)
               # print(f"Setting aperture radius to {r_opt} pixels.")
               # # Check r_opt relative to the FWHM of the filter:
               # if r_opt > 3 * fwhm:
               #      print("Large r_opt. Using PSF FWHM rather than curve of growth for photometry.")
               #      r_opt = 2.5 * fwhm

               # Load the filename from the config
               if "phot_cat_filename" in conf['parameters']['photometry']:
                    cat_filename = conf['parameters']['photometry']['phot_cat_filename']
               else:
                    cat_filename = f"{gal}_jwst_{band}_phot_cat_r{r_opt:4.2f}." + cat_filetype

               # Perform photometry with circular apertures
               if 'aperture_photometry' in steps:
                    print()
                    print(f"Performing photometry on {len(sources)} sources with aperture radius of {r_opt} pixels.")
                    apertures, catalog = compute_photometry(
                         # TODO this is important: do we want to use the background-subtracted image or the original image for this step?
                         data = use_image,
                         err = err,
                         header = header, 
                         gal = gal, 
                         band = band,
                         radius = r_opt,
                         sources = sources,
                         apcorr_method = apcorr_method,
                         # TODO put out_dir in cat_filename 
                         phot_cat_filename = cat_filename,
                         out_dir = local['out_dir'],
                         **conf['parameters']['photometry']
                    )

                    # print(f"Photometry complete. Catalog has {len(catalog)} sources.")

                    # Store the catalog in the catalogs dict
                    # print(catalog)
                    # print(catalog.colnames)
                    catalogs[gal][band] = catalog
               else:
                    phot_cat_path = os.path.join(local['out_dir'], cat_filename)
                    if os.path.exists(phot_cat_path):
                         catalogs[gal][band] = Table.read(phot_cat_path)
                         print(f"Loaded photometry catalog from {phot_cat_path}")
                    else:
                         print(f"Warning: aperture_photometry not requested and photometry catalog not found at {phot_cat_path}.")

               if "residual" in steps or "psffit" in steps:
                    print()
                    if "residual" in steps:
                         print(f"Computing residual image for {gal} at {band}...")
                    if "psffit" in steps:
                         print(f"Computing PSF fit for {gal} at {band}...")
                    if "psffit" in steps:
                         fittype=conf['parameters']['psffit']['fittype']
                    else:
                         fittype=None
                    fit_and_subtract(
                         datafile,
                         band=band,
                         srcfile=local['out_dir']+cat_filename, # source catalog to use for fitting 
                         file_root=local['out_dir'] + f"{gal}_{band}",
                         pixbinfactor=1.,
                         fittype=fittype, # "amp" or "amppos" or "ampwid" but here we want None to just create residual
                         doplot=True,
                         kflux='aperture_flux_mJy', # key name for app flux in input catalog
                         kdflux='tot_err_mJy',  # key name for app flux error in input catalog
                         kra='ra', # key name for RA in input catalog
                         kde='dec', # key name for Dec in input catalog
                         doregion=conf['parameters']['residual']['doregion'],
                         rd0=conf['parameters']['residual']['region_center'],
                         d=conf['parameters']['residual']['region_size'],
                         radius=r_opt
                    )

     return catalogs


catalogs = do_photometry(
               steps=steps, 
               targets=targets, 
               use_filter_fwhm=use_filter_fwhm,
               conf=conf
          )


exit()

