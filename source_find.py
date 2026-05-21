def run_source_finder(img, header, rms, finder='iraf', snr_threshold=3.0, fwhm=2.0,
     roundlo=-0.5, roundhi=0.5, sharplo=0.2, sharphi=1.0, brightest=50000):
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
               brightest=brightest,
          )
     elif finder == 'dao':
          # DAO can also use elliptical apertures.
          source_finder = DAOStarFinder(threshold=ths,
               fwhm=fwhm,
               roundlo=roundlo,
               roundhi=roundhi,
               sharplo=sharplo,
               sharphi=sharphi,
               brightest=brightest,
          )
    # elif finder == 'peaks':
     #      # find_peaks looks for local maxima above a specified threshold, 
     #      # and can use a box size to define the neighborhood for peak detection.
     #      # Requires a bit of extra work to get results in the same format as IRAFStarFinder/DAOStarFinder, 
     #      # and it doesn't calculate sharpness or roundness.
     #      # TODO: Add function converting find_peaks output to a table with xcentroid, ycentroid, flux, etc.
     #      source_finder = find_peaks(img, 
     #           threshold=ths, 
     #           box_size=fwhm*2,
     #           centroid_func=centroid_quadratic,
     #      )
     else:
          raise ValueError(f"Starfinder {finder} not recognized. Currently only 'iraf' supported.")
     #
     # TODO: Add functionality for source finding using a segmentation map (e.g., using 
     # detect_sources/SourceFinder) instead of IRAFStarFinder or DAOStarFinder.
     # Segmentation gets more information about the source shape rather than assuning it's point-like, 
     # giving a precise isophotal boundary. Useful for extended sources and crowded fields, 
     # but more computationally expensive and may require more tuning of parameters.
     # It is also better for source deblending.

     # Run the source finder
     sources = source_finder(img)
     print(f"Found {len(sources)} sources")
     print(sources.colnames)
     return sources