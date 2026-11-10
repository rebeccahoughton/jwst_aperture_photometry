import numpy as np
from astropy.stats import sigma_clipped_stats, sigma_clip


def radial_profile(data, center=None, bin_width=1.0, r_max=None,
                   statistic='mean', rms=False, sigma=3.0, maxiters=5,
                   return_counts=False):
    """
    Azimuthal radial profile of a 2D array.

    Parameters
    ----------
    data : 2D array
    center : (x, y) tuple, optional
        Centre in pixel coordinates (x = column, y = row). Defaults to the
        array centre ((nx-1)/2, (ny-1)/2).
    bin_width : float
        Radial bin width in pixels.
    r_max : float, optional
        Maximum radius. Defaults to the largest radius fully inside the array.
    statistic : {'mean', 'median', 'sigclip_mean'}
        Statistic computed per annulus.
    rms : bool
        If True, return the RMS profile sqrt(<x^2>) instead of the statistic.
        Combine with sigma-clipping by using rms_sigclip=True below.
    sigma, maxiters : sigma-clipping parameters (used by 'sigclip_mean' and
        sigma-clipped RMS).

    Returns
    -------
    r : ndarray       bin centres (pixels)
    prof : ndarray    profile values (NaN where a bin is empty)
    counts : ndarray  (optional) number of pixels used per bin
    """
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape

    if center is None:
        x0, y0 = (nx - 1) / 2.0, (ny - 1) / 2.0
    else:
        x0, y0 = center

    # Radius of every pixel (vectorised, computed once)
    y, x = np.indices(data.shape)
    r = np.hypot(x - x0, y - y0)

    if r_max is None:
        r_max = min(x0, y0, nx - 1 - x0, ny - 1 - y0)

    edges = np.arange(0.0, r_max + bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Bin index for each pixel; keep only pixels inside r_max and finite
    valid = np.isfinite(data) & (r < edges[-1])
    idx = np.digitize(r[valid], edges) - 1
    vals = data[valid]
    nbins = len(centres)

    # Group the pixel values by bin in one pass (sorted, then split)
    order = np.argsort(idx, kind='stable')
    idx_s, vals_s = idx[order], vals[order]
    bounds = np.searchsorted(idx_s, np.arange(nbins + 1))

    prof = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        seg = vals_s[bounds[i]:bounds[i + 1]]
        counts[i] = seg.size
        if seg.size == 0:
            continue

        if statistic == 'mean':
            prof[i] = np.sqrt(np.mean(seg**2)) if rms else np.mean(seg)

        elif statistic == 'median':
            prof[i] = np.sqrt(np.median(seg**2)) if rms else np.median(seg)

        elif statistic == 'sigclip_mean':
            clipped = sigma_clip(seg, sigma=sigma, maxiters=maxiters)
            good = clipped.compressed()
            if good.size:
                prof[i] = np.sqrt(np.mean(good**2)) if rms else np.mean(good)

        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sigclip_mean'")

    return (centres, prof, counts) if return_counts else (centres, prof)