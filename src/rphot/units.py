import warnings
import numpy as np
import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS, FITSFixedWarning


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
        print("WARNING: Input aperture sum must be in Jy/sr or MJy/sr for conversion to AB magnitudes.")
    
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
    # unless there is e.g. a PC matrix; 
    # so its better to just use wcs
    else:
        w=wcs.WCS(header) 
        return(w.proj_plane_pixel_area().to(u.sr).value)


    
def im2flux(sum,header):
    # convert from image units per pixel to flux units
    # returns value with units
    bunit = header['BUNIT']
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*OBSGEO.*",
            category=FITSFixedWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*datfix.*",
            category=FITSFixedWarning,
        )
        wcs = WCS(header)
    if bunit == 'MJy/sr':
        Mjy = sum * wcs.proj_plane_pixel_area() # proj_plane_pixel_area has units of deg2 so should cancel the sr but is astropy smart enough?
        return Mjy.to(u.mJy) 
    elif bunit == "erg/s/cm2/arcsec2":
        flux = sum * wcs.proj_plane_pixel_area() 
        return flux.to(u.erg / (u.s * u.cm**2))
    else:
        raise ValueError(f"Unsupported image units: {bunit}")


def flux2im(flux,header):
    # convert from flux to image units times pixels
    if not isinstance(flux, u.Quantity):
        raise TypeError("flux must be an astropy.units.Quantity")
    bunit = header['BUNIT']
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*OBSGEO.*",
            category=FITSFixedWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*datfix.*",
            category=FITSFixedWarning,
        )
        wcs = WCS(header)
    if bunit == 'MJy/sr':
        flux_mjy = flux.to(u.mJy) # will raise exception if not possible
        sum = flux_mjy / wcs.proj_plane_pixel_area()
        return sum.to(u.MJy / u.sr)
    elif bunit == "erg/s/cm2/arcsec2":
        flux = flux.to(u.erg / (u.s * u.cm**2)) # will raise exception if not possible
        sum = flux / wcs.proj_plane_pixel_area()
        return sum.to(u.erg / (u.s * u.cm**2 * u.arcsec**2))
    else:
        raise ValueError(f"Unsupported image units: {bunit}")