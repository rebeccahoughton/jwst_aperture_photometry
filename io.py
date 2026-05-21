import numpy as np
from astropy.io import fits
import glob



def convert_aperture_sum_Jy_per_sr_to_abmag(aperture_sum_jy_sr, header):
    '''Convert aperture sum in Jy/sr to AB magnitudes.
    Args:
        aperture_sum_jy_sr: aperture sum in Jy/sr (array-like)
        header: FITS header containing WCS information to get pixel area in steradians
    Returns:
        AB magnitudes (array-like)'''
    # here aperture_sum must be in Jy/sr
    pix_area_sr = get_pixarea_in_sr(header)
    fnu_jy = np.array(aperture_sum_jy_sr) * pix_area_sr
    fnu_jy = np.where(fnu_jy > 0, fnu_jy, np.nan)
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
        print("Warning: PIXAR_SR keyword not found in header. Attempting to compute pixel area from WCS information.")
        area_deg2 = np.abs(float(header['CDELT1']) * float(header['CDELT2']))
        if np.isfinite(area_deg2) and (area_deg2 > 0):
            return float((area_deg2 * u.deg**2).to(u.sr).value)
    elif 'CD1_1' in header:
        print("Warning: PIXAR_SR keyword not found in header. Attempting to compute pixel area from WCS information.")
        cd = np.array([[float(header['CD1_1']), float(header['CD1_2'])],
                        [float(header['CD2_1']), float(header['CD2_2'])]])
        area_deg2 = np.abs(np.linalg.det(cd))
        return float((area_deg2 * u.deg**2).to(u.sr).value)
    # And if we can't do either of those...
    else:
        raise ValueError("could not get pixel area in steradians from header/WCS")



def open_jwst(path, gal, dir, band, level, mosaic_ext="*i2d_anchor.fits", get_coverage=True):
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
    files = glob.glob(f"{path}/{gal}*{band}*{mosaic_ext}")
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