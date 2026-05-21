import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry


def subtract_bkg(img, box_size=(500,500), filter_size=(3, 3), bkg_estimator=MedianBackground(), sigma=3.0, coverage_mask=None):
     """Estimate and subtract background from image using Background2D.
     Args:
          img: 2D array of image data
          box_size: size of boxes for background estimation (in pixels)
          filter_size: size of median filter to apply to background (in pixels)
          bkg_estimator: background estimator to use (default is MedianBackground())
          sigma: sigma for sigma clipping (default is 3.0)
          coverage_mask: boolean array where True indicates pixels to exclude from background estimation 
                         (e.g., due to low coverage or bad data). If None, no mask is applied.
    Returns:
          img_sub: background-subtracted image
          bkg_mean: mean background level (in same units as img)
          bkg_rms: background RMS (in same units as img)"""
     
     # estimate background
     # TODO: need to include valid mask based on weight image or other metric
     sigma_clip = SigmaClip(sigma=sigma)

     if coverage_mask is None:
          print(f"creating coverage mask")
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




def get_optimal_aperture(data, sources, max_r=32, brightest=50, frac=0.95, plot=True):
     """Find the optimal aperture radius to use for the photometry. 
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