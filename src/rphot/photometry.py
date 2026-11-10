import warnings
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from sys import exit
from astropy.wcs import WCS, FITSFixedWarning
from astropy.stats import SigmaClip
from astropy.visualization import ImageNormalize, LogStretch

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry

from rphot.units import convert_aperture_sum_Jy_per_sr_to_abmag, im2flux
from rphot.apcorr import get_apcorr_from_psf, get_apcorr_params


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

    # compute median normalized flux
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
        apcorr_params = None,
        local_bkg_subtract=False,
        out_dir="./",
        psf_dir="./",
        crds_dir="./",
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
         write: if True, write catalog to out_dir with name {gal}_{band}_cat.fits
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
        # this is the use case of using a previous photometry catalog for forced photometry
        # on a new image (you put that catalog in "find_cat_filename")
        # these cases could be generalized to find a generic flux measurement in the file 
        bkeys = ['peak_value', 'aperture_flux', 'flux_F336W']
        for bk in bkeys:
            if bk in sources.colnames:
                kbrightness = bk
                break

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
        apcorr = get_apcorr_from_psf(band, radius, radius_sky_in, radius_sky_out, psf_dir)
        print(f"Using PSF-based aperture correction factor of {apcorr} for radius {radius} pixels.")
    elif apcorr_method != None:
        # Get aperture correction parameters from CRDS file
        radius, radius_sky_in, radius_sky_out, apcorr = get_apcorr_params(crds_dir, band, inst='NIRCam', **apcorr_params)
        print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels.")

    # Do aperture photometry
    print()
    print(f"Doing aperture photometry for {len(sources)} sources...")
    kx = 'xcentroid'; ky = 'ycentroid'
    if kx not in sources.colnames or ky not in sources.colnames:
        kx = 'x_center'; ky = 'y_center'
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

    # Subtract background from aperture sum
    phot_full['aperture_flux'] = im2flux( phot_full['aperture_sum'], header )
    if local_bkg_subtract:
        phot_full['bkg_median'] = bkg_median
        phot_full['bkg_flux'] = im2flux( bkg_median * aper_stats.sum_aper_area.value, header )
        phot_full['aperture_flux'] -= phot_full['bkg_flux']
    
    # Copy source-finder morphology columns
    if 'flux' in sources.colnames:  # it won't be there for findpeaks method.  TODO could be added in find step
        phot_full['finder_flux'] = im2flux( sources['flux'], header )
    if 'sharpness' in sources.colnames:
        phot_full['finder_sharpness'] = np.asarray(sources['sharpness'])
    if 'roundness' in sources.colnames:
        phot_full['finder_roundness'] = np.asarray(sources['roundness'])
    if 'mag' in sources.colnames:
        phot_full['finder_mag'] = np.asarray(sources['mag'])
    if 'peak' in sources.colnames:
        phot_full['finder_peak'] = sources['peak'] # keep image units on "peak"
    elif 'peak_value' in sources.colnames: 
        phot_full['finder_peak'] = sources['peak_value']   
    if 'reject' in sources.colnames:
        phot_full['finder_centroid_fail'] = sources['reject']
    if 'id' in sources.colnames:
        phot_full['id'] = sources['id']
    if 'ID_phangs' in sources.colnames:
        sources['id'] = sources['ID_phangs']
    if 'ID_source' in sources.colnames:
        sources['id'] = sources['ID_source']
    
    # Include ra, dec
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*OBSGEO.*",
            category=FITSFixedWarning,
        )
        wcs = WCS(header)
    if "x_center" in phot_full.colnames:
        ra, dec = wcs.all_pix2world(phot_full["x_center"], phot_full["y_center"], 0)
    else:
        ra, dec = wcs.all_pix2world(phot_full["xcenter"], phot_full["ycenter"], 0)
    pixscale = wcs.proj_plane_pixel_scales()[0].to("arcsec")
    phot_full["ra"]  = np.round(ra ,8)
    phot_full["dec"] = np.round(dec,8)

    # Convert flux from the source finder in table (converted to AB magnitudes)
    if 'finder_flux' in phot_full.colnames:
        phot_full['finder_flux_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['finder_flux'], header=header)
    # Aperture sum from circular aperture photometry (converted to AB magnitudes)
    phot_full['aperture_sum_abmag'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header)

    if apcorr_method != None:
        if apcorr.unit.is_equivalent(u.dimensionless_unscaled):
            phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'] * apcorr, header=header)
        elif apcorr.unit.is_equivalent(u.mag):
            phot_full['aperture_sum_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_sum'], header=header) + apcorr.value
        # TODO do we need to multiply the error by the apcorr? Jimena did not.
        if 'aperture_flux' in phot_full.colnames:
            phot_full['aperture_flux_apcorr'] = phot_full['aperture_flux'] * apcorr if apcorr.unit.is_equivalent(u.dimensionless_unscaled) else phot_full['aperture_flux'] + apcorr.value
            phot_full['aperture_flux_abmag_apcorr'] = convert_aperture_sum_Jy_per_sr_to_abmag(phot_full['aperture_flux_apcorr'], header=header) 

    # special step for the continuum subtracted PAH bands to filter some of the poor subtractions:
    if "pah" in band.lower():
        negative_threshold = -0.5
        z_neg = np.where(phot_full['aperture_min'] < negative_threshold)[0]
        phot_full['aperture_sum_err'][z_neg] *= 3
        if len(z_neg) > 0:
            print(f"Found {len(z_neg)} sources with aperture_min < {negative_threshold}. Increased their aperture_sum_err by a factor of 3.")

    # Error on the flux due to background estimation uncertainty.
    # The worst case is that of structured background, where the error scales with the area of the aperture.  
    # The best case is that of unstructured background, where the error scales with the square root of the area of the aperture.  
    bkg_err_scalefactor = np.sqrt(0.5*np.pi / bkg_stats.sum_aper_area.value)  # scale factor for background error based on area of annulus 

    phot_full['bkg_err'] = im2flux( bkg_stats.std * aper_stats.sum_aper_area.value, header )
    phot_full['poisson_err'] = im2flux( phot_full['aperture_sum_err'], header )
 
    # TODO: Is there a better way to do this than a list?  add to the filter table!
    if band.lower()=='f335m' or band.lower()=='f770w' or band.lower()=='f1000w' or band.lower()=='f1130w' or band.lower()=='f2100w' or "pah" in band.lower():
        phot_full['total_err'] = np.sqrt(phot_full['poisson_err']**2 + phot_full['bkg_err']**2 )
    elif band.lower()=='f200w' or band.lower()=='f300m' or band.lower()=='f360m' or band.lower()=='f444w':
        phot_full['total_err'] = np.sqrt(phot_full['poisson_err']**2 + (phot_full['bkg_err']*bkg_err_scalefactor)**2)
    else:
        print(f"Band {band} not recognized for error calculation. Setting total_err to max possible, sqrt(poisson_err**2 + bkg_err**2).")
        phot_full['total_err'] = np.sqrt(phot_full['poisson_err']**2 + phot_full['bkg_err']**2)


    phot_full['total_SNR'] = phot_full['aperture_flux'] / phot_full['total_err']
    phot_full['poisson_SNR'] = phot_full['aperture_flux'] / phot_full['poisson_err']
    phot_full['bkg_SNR'] = phot_full['aperture_flux'] / phot_full['bkg_err']

    phot_full.meta['pixel_scale'] = pixscale
    phot_full.meta['ap_radius_pix'] = radius
    phot_full.meta['ap_radius_arcsec'] = radius * phot_full.meta['pixel_scale']
    phot_full.meta['sky_in_radius_pix'] = radius_sky_in
    phot_full.meta['sky_in_radius_arcsec'] = radius_sky_in * phot_full.meta['pixel_scale']
    phot_full.meta['sky_out_radius_pix'] = radius_sky_out
    phot_full.meta['sky_out_radius_arcsec'] = radius_sky_out * phot_full.meta['pixel_scale']
    phot_full.meta['apcorr'] = apcorr

    # Write the catalog if requested
    if write:
        if phot_cat_filename is None:
            phot_cat_filename = f"{gal}_{band}_phot_cat_r{radius:4.2f}." + cat_filetype
        print(f"Writing catalog to {out_dir + "/" + phot_cat_filename}")
        # make a plain table to avoid serialized columns, which break TopCat
        #Table(phot_full).write(out_dir + phot_cat_filename, overwrite=overwrite)
        phot_full.write(out_dir + "/" + phot_cat_filename, overwrite=overwrite)

    if doplot:
        # plot photometry results
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(phot_full['aperture_flux'], phot_full['bkg_err']/phot_full['aperture_flux'], 'o', markersize=1, alpha=0.5, label="bg")
        ax.plot(phot_full['aperture_flux'], phot_full['poisson_err']/phot_full['aperture_flux'], 'o', markersize=1, alpha=0.5, label="poisson")
        ax.legend(loc="best",prop={"size":8})
        ax.set_xlabel(f"Flux ({phot_full['aperture_flux'].unit})")
        ax.set_ylabel("Error/Flux")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(out_dir+f"/{gal}_{band}_appflux_dflux.png", dpi=300)
        plt.close(fig)

        # Plot the image with significant sources 
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        norm = ImageNormalize(vmin=np.nanpercentile(data.value, 25.00), 
                              vmax=np.nanpercentile(data.value, 99.99), 
                              stretch=LogStretch())
        ax.imshow(data.value, origin='lower', cmap='inferno', norm=norm)
        ax.set_title(f"{gal.upper()} {band.upper()}")
        # z=np.where(phot_full['aperture_flux']/phot_full['poisson_err']>5)[0]
        # ax.scatter(sources['xcentroid'][z], sources['ycentroid'][z], s=10, edgecolor='cyan', facecolor='none', lw=0.5, alpha=0.3,label="Poisson SNR>5")
        label_low_snr_values = False  # Set to True to label each red low-SNR source with its SNR value
        err_threshold = 1.5
        snr = phot_full['aperture_flux'] / phot_full['total_err']
        hi_label_added = False
        lo_label_added = False
        for idx in range(len(phot_full)):
            if "x_center" in phot_full.colnames:
                x = phot_full['x_center'][idx]
                y = phot_full['y_center'][idx]
            else:
                x = phot_full['xcenter'][idx]
                y = phot_full['ycenter'][idx]
            linewidth = 1
            alpha = 0.7     
            if snr[idx] > err_threshold:
                edgecolor = 'k'
                label = None if hi_label_added else f"SNR > {err_threshold}"
                hi_label_added = True
            elif snr[idx] >0 and phot_full['aperture_flux'][idx]/phot_full['poisson_err'][idx]>5:
                edgecolor = 'g'
                label = None if lo_label_added else f"SNR < {err_threshold}"
                lo_label_added = True
            else:
                continue
            circ = plt.Circle((x, y), radius=radius, fill=False, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, label=label)
            ax.add_patch(circ)
            if snr[idx] < err_threshold and label_low_snr_values:
                #ax.text(x + radius * 1.2, y + radius * 1.2, f"{snr[idx]:.1f}", color='g', fontsize=8, ha='left', va='bottom')
                ax.text(x + radius * 1.2, y + radius * 1.2, f"{phot_full['aperture_flux'][idx]/phot_full['poisson_err'][idx]:.1f}", color='g', fontsize=8, ha='left', va='bottom')
                  
        ax.legend(loc="best",prop={"size":8})
        im = ax.images[0]
        plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
        plt.savefig(out_dir+f"/{gal}_{band}_significant_sources.png", dpi=600)
        plt.close(fig)


        # TODO put in config file
        min_halfcut_asec = 0.7 * u.arcsec
        # Convert minimum half-cutout size from arcsec to pixels
        min_halfcut_pix = (min_halfcut_asec / pixscale).decompose().value
    
        # Make the 6x6 plot of source cutouts using sources from the aperture corrected catalog
        # nonan=np.where(np.isfinite(phot_full['aperture_flux_mJy']))[0]
        # brightest_sources_apcorr = phot_full[nonan][np.argsort(phot_full['aperture_flux_mJy'][nonan])[::-1]][:36]
        # instead of re-sorting, leave these in the order from the finder which sorted them from brightest
        # peak to lower peak values - this allows forced photometry to have the same 36 sources in each band
        brightest_sources_apcorr = phot_full[:36]
        # brightest_sources_apcorr = phot_full[nonan][:36]
        fig, axes = plt.subplots(6, 6, figsize=(10, 10), 
                                 gridspec_kw={'left':0.02,'bottom':0.02,'top':0.98,'right':0.98,
                                              'wspace': 0.01, 'hspace': 0.01})
        for i, (ax, row) in enumerate(zip(axes.flatten(), brightest_sources_apcorr)):
            if "x_center" in row.colnames:
                x, y = row['x_center'], row['y_center']
            else:
                x, y = row['xcenter'], row['ycenter']   
            cutout_size = np.max([radius*4, radius_sky_out*1.1, min_halfcut_pix]) # actually, half-cutout size
            x_min, x_max = int(x - cutout_size), int(x + cutout_size)
            y_min, y_max = int(y - cutout_size), int(y + cutout_size)
            cutout = data[y_min:y_max, x_min:x_max]
            norm_cutout = ImageNormalize(vmin=np.nanpercentile(cutout.value, 0), 
                                         vmax=np.nanpercentile(cutout.value, 100),
                                         # stretch=SqrtStretch()
            )
            ax.imshow(data.value, origin='lower', cmap='inferno', norm=norm_cutout)
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
            if row['aperture_flux'].unit == u.mJy:
                ax.text(0.5, 0.9, f"{row['aperture_flux'].value*1000:.1f}uJy ({row['bkg_flux'].value*1000:.2f})", color='white', fontsize=8, ha='center', va='center', transform=ax.transAxes)               
            else:
                ax.text(0.5, 0.9, f"{row['aperture_flux']:.1f}", color='white', fontsize=8, ha='center', va='center', transform=ax.transAxes)
            #ax.text(0.08, 0.93, f"{i+1}", color='cyan', fontsize=8, ha='center', va='center', transform=ax.transAxes)
            #ax.text(0.08, 0.05, f"{x:.0f},{y:.0f}", color='cyan', fontsize=8, ha='left', va='center', transform=ax.transAxes)
            ax.text(0.08, 0.05, f"{row['ra']:.4f},{row['dec']:.4f}", color='cyan', fontsize=6, ha='left', va='center', transform=ax.transAxes)
            # Select all sources in the catalog that are within the cutout region
            if 'x_center' in phot_full.colnames:
                sources_in_cutout = phot_full[(phot_full['x_center'] > x_min) & (phot_full['x_center'] < x_max) & (phot_full['y_center'] > y_min) & (phot_full['y_center'] < y_max)]
                ax.scatter(sources_in_cutout['x_center'], sources_in_cutout['y_center'], s=50, edgecolor='cyan', facecolor='none', lw=1.0, alpha=0.5)
            else:
                sources_in_cutout = phot_full[(phot_full['xcenter'] > x_min) & (phot_full['xcenter'] < x_max) & (phot_full['ycenter'] > y_min) & (phot_full['ycenter'] < y_max)]
                ax.scatter(sources_in_cutout['xcenter'], sources_in_cutout['ycenter'], s=50, edgecolor='cyan', facecolor='none', lw=1.0, alpha=0.5)
        plt.savefig(out_dir+f"/{gal}_{band}_cutouts_brightest_r{radius:4.2f}.png", dpi=400)
        plt.close(fig)




    return apertures, phot_full