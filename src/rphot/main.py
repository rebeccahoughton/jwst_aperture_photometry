import os
import warnings
import numpy as np
from sys import exit
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table, QTable

from rphot.io import get_file, load_filter_data, load_config, open_jwst
from rphot.units import get_pixarea_in_sr
from rphot.catalog import combine_by_id
from rphot.background import subtract_bkg
from rphot.detect import run_source_finder
from rphot.photometry import compute_photometry, get_optimal_aperture
from rphot.residuals import residual, fit_and_subtract


# Directories
config_file = 'config/singleband.toml'     # Photometry parameters
config_file = 'config/config_hst_force.toml'     # Photometry parameters
local_file = 'config/va.toml'       # Paths to directories

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))), config_file)
local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))), local_file)

# TODO make flux limits unit-aware
lowfluxlim = 1e-5 # below this its a nondetection

extensions_to_try = ['SCI', 'CONTSUB','PRIMARY']

# Unpack the parameters from the config file
conf = load_config(config_path)
local = load_config(local_path)

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

use_filter_fwhm = phot_params['use_filter_fwhm']

if use_filter_fwhm and "r_opt" in steps:
    raise ValueError("use_filter_fwhm=True cannot be used with r_opt step")

# Unpack local directories
jwst_dir = local['jwst_dir']
out_dir = local['out_dir']
psf_dir = local['psf_dir']
crds_dir = local['crds_dir']
cat_path = local['out_dir']  
cat_filetype = conf['cat_filetype']

# TODO: add cat_path to local.toml if we want to load in an external catalog for photometry instead of running a source finder.

# Get properties for the filter being used
filter_fwhm_pix, filter_wave = load_filter_data()

# Check that input data directory exists
if not os.path.exists(jwst_dir):
    raise FileNotFoundError(f"JWST data directory {jwst_dir} does not exist. Please check the path in the config file.")
    exit()
# Check that out_dir exists
if not os.path.exists(out_dir):
    raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
    exit()


# Run
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
            # get distance from the galaxy sample table (override with config file)
            sample_tab_path = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))), "data/phangs_sample_table_v1p6.ecsv")
            sample_tab = Table.read(sample_tab_path)
            dist_Mpc = sample_tab[sample_tab['name'] == gal]['dist'][0]
            if 'dist_Mpc' in conf['parameters']['bkg_subtract']:
                dist_Mpc = conf['parameters']['bkg_subtract']['dist_Mpc']
            conf['parameters']['bkg_subtract']['dist_Mpc'] = dist_Mpc
            print(f"Distance for {gal}: {dist_Mpc} Mpc")

            pix_scale = get_pixarea_in_sr(header) ** 0.5 * (180/np.pi) * 3600  # arcsec/pixel

            # Subtract background 
            if 'subtract_bkg' in steps:
                print()
                print(f"Subtracting background for {gal} at {band}...")
                if 'box_size_pix' not in conf['parameters']['bkg_subtract']:
                    # Convert box size from pc to pixels using the pixel scale from the header
                    box_size_pc = conf['parameters']['bkg_subtract']['box_size_pc']
                    box_size_pix = int(box_size_pc * 206265 / (pix_scale * dist_Mpc * 1e6 ))
                    conf['parameters']['bkg_subtract']['box_size_pix'] = box_size_pix

                if 'filter_size_pix' not in conf['parameters']['bkg_subtract']:
                    # Convert filter size from pc to pixels using the pixel scale from the header
                    filter_size_pc = conf['parameters']['bkg_subtract']['filter_size_pc']
                    filter_size_pix = int(filter_size_pc * 206265/ (pix_scale * dist_Mpc * 1e6 ))
                    conf['parameters']['bkg_subtract']['filter_size_pix'] = filter_size_pix

                img_sub, bkg_mean, bkg_rms, bkg_background = subtract_bkg(
                    image_path=datafile,
                    gal=gal,
                    band=band,
                    out_dir=local['out_dir'],
                    **conf['parameters']['bkg_subtract']
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
                cat_filename = f"{gal}_{band}_find_cat." + cat_filetype

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
                    **conf,
                    **local,
                )
            
            else:
                print()
                print(f"Importing sources from existing catalog for {gal} at {band}...")
                # Load the existing source catalog
                sources = QTable.read(cat_path + cat_filename)
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
                elif ('ra' in sources.colnames and 'dec' in sources.colnames) or \
                    ('raj2000' in sources.colnames and 'dej2000' in sources.colnames):
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*OBSGEO.*",
                            category=FITSFixedWarning,
                        )
                        current_wcs = WCS(header)
                    if 'ra' in sources.colnames and 'dec' in sources.colnames:
                        xcentroid, ycentroid = current_wcs.all_world2pix(
                            sources['ra'], sources['dec'], 0
                        )
                    else:
                        xcentroid, ycentroid = current_wcs.all_world2pix(
                            sources['raj2000'], sources['dej2000'], 0
                        )
                    sources['xcentroid'] = xcentroid
                    sources['ycentroid'] = ycentroid

                if 'ID_phangs' in sources.colnames:
                    sources['id'] = sources['ID_phangs']

                # Checks that the colnames include x_centroid, y_centroid, flux, sharpness, roundness, mag, peak, etc. 
                # and print a warning if any are missing
                required_cols = ['xcentroid', 'ycentroid', 'flux']
                x_to_search_for = ['xcentroid', 'x_center', 'x_centroid', 'x_center']
                y_to_search_for = ['ycentroid', 'y_center', 'y_centroid', 'y_center']
                # Cycle through
                # TODO this looks like x_to_search_for was not fully wired into the ifelse here:
                for col in required_cols:
                    if col not in sources.colnames:
                        # Check whether there is a x_center and y_center column instead of x_centroid and y_centroid, and if so, rename them
                        if col == 'xcentroid' and 'x_center' in sources.colnames:
                            sources['xcentroid'] = sources['x_center']
                        elif col == 'ycentroid' and 'y_center' in sources.colnames:
                            sources['ycentroid'] = sources['y_center']
                        else:
                            print(f"Warning: Column '{col}' is missing from the external catalog."
                                f"\n Please make sure the catalog has the required columns: {required_cols}.")

            # Either get the optimum radius based on curve of growth...
            if 'r_opt' in steps:
                print(f"Computing optimal aperture for photometry...")
                if 'subtract_bkg' not in steps:
                    print("Warning: r_opt step is being run without background subtraction. This may affect the results.")
                r_opt = get_optimal_aperture(
                    data = use_image,
                    sources = sources,
                    **conf['parameters']['r_opt'],
                    **local,
                )
            else:
                fwhm2rad = conf['parameters']['photometry'].get('fwhm2rad', 2.5)
                if use_filter_fwhm:
                    r_opt = filter_fwhm_pix[band.upper()] * fwhm2rad
                else:
                    r_opt = conf['parameters']['photometry']['aperture_radius_arcsec']
                    if type(r_opt) == list:
                        if len(r_opt) != len(bands):
                            raise ValueError("Length of r_opt list must match the number of bands.")
                        r_opt = r_opt[bands.index(band)]
                    r_opt /= pix_scale  # Convert from arcseconds to pixels
                    print(f"Using fixed aperture radius of {r_opt*pix_scale} arcsec.")

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
                phot_cat_filename = conf['parameters']['photometry']['phot_cat_filename']
            else:
                phot_cat_filename = f"{gal}_{band}_phot_cat_r{r_opt:4.2f}." + cat_filetype
                if 'phot_cat_extrastring' in phot_params:
                    if phot_params['phot_cat_extrastring'] is not None and len(phot_params['phot_cat_extrastring']) > 0:
                        phot_cat_filename = f"{gal}_{band}_phot_cat_{phot_params['phot_cat_extrastring']}_r{r_opt:4.2f}." + cat_filetype


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
                    phot_cat_filename = phot_cat_filename,
                    apcorr_params = conf['parameters']['apcorr'],
                    **conf['parameters']['photometry'],
                    **local,
                )

                # print(f"Photometry complete. Catalog has {len(catalog)} sources.")

                # Store the catalog in the catalogs dict
                catalogs[gal][band] = catalog

                # ds9 region file for visualizing sources
                ds9reg=open(local['out_dir']+'/'+'.'.join(phot_cat_filename.split('.')[:-1])+".reg","w")
                ds9reg.write("fk5\n")
                r_opt_asec = r_opt * header['CDELT2'] * 3600.  # convert pixels to arcseconds
                for src in catalog:
                    # TODO make this unit-aware
                    # if src['aperture_flux'].value > lowfluxlim:
                    if (src['aperture_flux'].value > lowfluxlim)* \
                        ((src['aperture_flux']/src['total_err'])>1):
                        if 'reject' in catalog.colnames and catalog['reject'][src.index] != "keep":
                            ds9reg.write(f'circle({src['ra']},{src['dec']},{r_opt_asec}") # color=red\n')
                        else:
                            ds9reg.write(f'circle({src['ra']},{src['dec']},{r_opt_asec}") # color=blue\n')
                    else:
                        ds9reg.write(f'circle({src['ra']},{src['dec']},{r_opt_asec/3}") # color=yellow dash=1 width=1\n')
                ds9reg.close()


            else:
                phot_cat_path = local['out_dir'] + "/" + phot_cat_filename
                if os.path.exists(phot_cat_path):
                    catalogs[gal][band] = QTable.read(phot_cat_path)
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
                catalog = fit_and_subtract(
                    datafile,
                    gal=gal,
                    band=band,
                    srcfile=local['out_dir']+'/'+phot_cat_filename, # source catalog to use for fitting 
                    pixbinfactor=1.,
                    fittype=fittype, # "amp" or "amppos" or "ampwid" but here we want None to just create residual
                    doplot=True,
                    kflux='aperture_flux', # key name for app flux in input catalog
                    kdflux='total_err',  # key name for app flux error in input catalog
                    kra='ra', # key name for RA in input catalog
                    kde='dec', # key name for Dec in input catalog
                    doregion=conf['doregion'],
                    rd0=conf['region_center'],
                    d  =conf['region_size'],
                    radius=r_opt,
                    psf_dir=local['psf_dir'],
                    cat_filetype = conf['cat_filetype'],
                    local=local, config=conf
                )

                # Store (replace) the catalog in the catalogs dict
                catalogs[gal][band] = catalog


    return catalogs


catalogs = do_photometry(
            steps=steps, 
            targets=targets, 
            use_filter_fwhm=use_filter_fwhm,
            conf=conf
        )

# combine catalogs by ID for each galaxy across bands
for gal in catalogs:
    if len(catalogs[gal]) > 1:
        combined_catalog = combine_by_id(catalogs[gal], 
                                          keys_to_discard=['aperture_sum', 'aperture_sum_err', 'aperture_sum_abmag', 'aperture_sum_abmag_err', 'x_center', 'y_center',
                                                           'psffit_ra', 'psffit_dec'],
                                          common_keys=['id','ra','dec'])
        combined_catalog.write(local['out_dir'] + f"/{gal}_combined." + cat_filetype, overwrite=True)