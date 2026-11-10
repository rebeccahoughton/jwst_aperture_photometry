import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LogStretch

from photutils.detection import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.centroids import centroid_quadratic

from sys import exit


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
        doregion=False,
        region_center=None,
        region_size=None,
        cat_filetype="fits",
        out_dir="./",
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
    print(out_dir)
    
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
        # RI this does end up keeping sources on the edges which we don't want, but it does save a blobby source
        # even worse ,it includes a LOT of chaff, so only keep the super-bright ones
        # TODO make the cut unit-aware
        #----------------------------------------
        z=np.where((np.isnan(sources['x_centroid']))*(sources['peak_value'].value>5))[0]
        sources['reject'] = ["keep"]*len(sources)
        if len(z)>0:
            sources['reject'][z] = "centroid_fail"
            sources['x_centroid'][z]=sources['x_peak'][z]        
            sources['y_centroid'][z]=sources['y_peak'][z]
        z=np.where((np.isnan(sources['x_centroid']))*(sources['peak_value'].value<5))[0]
        if len(z)>0:
            print(f"Warning: discarding {len(z)} sources with NaN centroids")
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

    # sources is a QTable and radec will have u.deg
    sources['ra'] = sk.ra
    sources['dec'] = sk.dec
    sources['ra_centroid'] = sk.ra
    sources['dec_centroid'] = sk.dec

    if doplot:
        # Plot the image with sources 
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        norm = ImageNormalize(vmin=np.nanpercentile(img.value, 20.00), 
                              vmax=np.nanpercentile(img.value, 100.), 
                              stretch=LogStretch())
        # matplotlib tries to match the units of the color bar with those of the image, which can be error-prone
        ax.imshow(img.value, origin='lower', cmap='viridis', norm=norm)
        ax.set_title(f"{gal.upper()} {band.upper()} mosaic")
        ax.scatter(sources['xcentroid'], sources['ycentroid'], s=10, edgecolor='cyan', facecolor='none', lw=0.5, alpha=0.2)
        im = ax.images[0]
        plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
        plt.savefig(out_dir+f"/{gal}_{band}_source_finder_{finder}.png", dpi=300)
        # zoom in on a region
        if doregion:
            x0,y0=wcs_obj.world_to_pixel(SkyCoord(region_center[0]*u.deg, region_center[1]*u.deg))
            zoom_size=region_size/wcs_obj.proj_plane_pixel_scales()[0].to(u.deg).value
        else:
            x0=np.mean(plt.xlim())
            y0=np.mean(plt.ylim())
            zoom_size = 100  # size of the zoomed-in region in pixels
        ax.set_xlim(x0 - zoom_size/2, x0 + zoom_size/2)
        ax.set_ylim(y0 - zoom_size/2, y0 + zoom_size/2)
        ax.set_title(f"{gal.upper()} {band.upper()} mosaic, S/N threshold = {snr_threshold}")
        ax.scatter(sources['xcentroid'], sources['ycentroid'], s=20, edgecolor='cyan', facecolor='none', lw=2, alpha=0.8)
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
            cat_name = f"{gal}_{band}_find_cat." + cat_filetype
        print(f"Writing catalog to {out_dir + '/' + cat_name}")
        sources.write(out_dir + '/' + cat_name, overwrite=overwrite)

    return sources