import numpy as np
import astropy.units as u
from scipy.spatial import cKDTree
from astropy.table import hstack, join


def match(
    catalog1, 
    catalog2, 
    npix=2, 
    keys=['catalog1', 'catalog2']):
    
    coords2 = np.array([catalog2['x_center'], catalog2['y_center']]).T
    coords1 = np.array([catalog1['x_center'], catalog1['y_center']]).T
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



def combine_by_id( 
            catalogs,
            keys_to_discard=[],
            common_keys=['id'] ):  
    outcat = {}
    for iband,band in enumerate(catalogs):
        catalog = catalogs[band]
        kk = catalog.colnames.copy()
        for key in kk:
            if key in keys_to_discard:
                del(catalog[key])
            elif key in common_keys:
                continue
            else:
                catalog.rename_column(key, f"{band}_{key}")
        kk = list(catalog.meta.keys())
        for key in kk:
            # complex meta key types can cause issues if saved to fits 
            if type(catalog.meta[key]) in [int, float, str, bool, u.quantity.Quantity] and key not in ["date", "aperture_photometry_args"]:
                catalog.meta[f"{band}_{key}"] = catalog.meta[key]
            del(catalog.meta[key])

        if iband == 0:
            outcat= catalog # metadata transfers here
        else:
            # really should not need outer since every cat should have all ids, but in case...
            outcat = join(outcat, catalog, keys=common_keys, join_type='outer') 
            for key in catalog.meta.keys():                    
                outcat.meta[key] = catalog.meta[key]

    return outcat   