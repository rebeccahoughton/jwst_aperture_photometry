import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization import ImageNormalize, LogStretch
from photutils.background import Background2D, MedianBackground, SExtractorBackground

from rphot.io import open_jwst

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
        out_dir="./",
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

    rms_map = bkg.background_rms
    valid_rms = (~coverage_mask) & np.isfinite(rms_map) & (rms_map > 0)
    # print(f"bkg array {bkg.background}")
    bkg_rms = np.nanmedian(rms_map[valid_rms]) if np.any(valid_rms) else np.nan
    bkg_mean = np.nanmean(np.asarray(bkg.background, dtype=float)[~coverage_mask])

    # print(f"bkg array {bkg.background}")
    if image_path is not None:
        # background_path = out_dir + "/" + os.path.basename(image_path).replace(".fits", "_background.fits")
        background_path = out_dir + f"/{gal}_{band}_background.fits"
        out_header = header.copy() if header is not None else fits.Header()
        hdu = fits.PrimaryHDU(data=np.asarray(bkg.background, dtype=float), header=out_header)
        hdu.writeto(background_path, overwrite=True)
        print(f"Saved background map to {background_path}")

    # threshold_img = snr_threshold * bkg.background_rms
    img_sub = img - bkg.background

    bgsub_path = out_dir + f"/{gal}_{band}_bgsub.fits"
    out_header = header.copy() if header is not None else fits.Header()
    hdu = fits.PrimaryHDU(data=np.asarray(img_sub, dtype=float), header=out_header)
    hdu.writeto(bgsub_path, overwrite=True)
    print(f"Saved background-subtracted image to {bgsub_path}")

    if doplot:
        # Plot the image, background, and background-subtracted image
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        norm = ImageNormalize(vmin=np.nanpercentile(img.value, 25.00),
                              vmax=np.nanpercentile(img.value, 99.99),
                              stretch=LogStretch())
        ax[0].imshow(img.value, origin='lower', cmap='inferno', norm=norm)
        ax[0].set_title(f"{gal.upper()} {band.upper()} mosaic")
        # TODO: gal and band as global properties
        ax[1].imshow(bkg.background.value, origin='lower', cmap='inferno')
        ax[1].set_title("Estimated background")
        img_sub = img - bkg.background
        norm_sub = ImageNormalize(vmin=np.nanpercentile(img_sub.value, 25.00),
                                  vmax=np.nanpercentile(img_sub.value, 99.99),
                                  stretch=LogStretch())
        ax[2].imshow(img_sub.value, origin='lower', cmap='inferno', norm=norm_sub)
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
        out_dir="./",
        **kwargs):
    """Create a cached background FITS file if needed and subtract it from the image."""
    background_path = out_dir + f"/{gal}_{band}_background.fits"
    
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
            out_dir=out_dir,
            header=header,
            **kwargs,
        )
    else:
        print(f"Background file found on disk for {image_path}.")

    img, err, snr_map, coverage_mask, header = open_jwst(image_path)
    bkg_background = fits.getdata(background_path) * img.unit
    bkg_mean = np.nanmean(bkg_background)
    bkg_rms = np.nanstd(bkg_background)
    img_sub = img - bkg_background

    return img_sub, bkg_mean, bkg_rms, bkg_background