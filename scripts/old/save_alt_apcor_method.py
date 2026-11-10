










# ---------------------------------------------------------------------------------------------------------

# Alternative approach using standardised aperture corrections from the JWST CRDS.
path_to_crds = "/nexus/posix0/MIA-astro-env/eschinner/jgonzalez/jwst_pipeline/crds_cache/jwst_ops/references/jwst/" + 'nircam' + "/"

# Get the apcorr file using glob
apcorr_files = glob.glob(path_to_crds + f"*apcorr*")
if len(apcorr_files) == 0:
    raise FileNotFoundError(f"No apcorr files found for {band} in {inst} at {path_to_crds}")
else:
    print(f"Found apcorr files: {apcorr_files}")

# Load the file
apcorr_data = fits.getdata(apcorr_files[0], ext=1)
print(f"APCORR data columns: {apcorr_data.columns.names}")

# Print all the unique filters in the apcorr file
print("Unique eefraction values:", np.unique(apcorr_data['eefraction']))

# Get data for a specific eefraction
# The eefraction is the fraction of the total flux that is enclosed within the aperture radius.
eefraction_value = 0.70
row = apcorr_data[apcorr_data['eefraction'] == eefraction_value]

# Limit to a specific filter 
row = row[(row['filter'] == band.upper())]

# Extract values
wcs_apcorr = WCS(header)
radius = row['radius'][0]   # in pixels
sky_in = row['skyin'][0]    # in pixels
sky_out = row['skyout'][0]  # in pixels
apcorr = row['apcorr'][0]   # factor to multiply enclosed flux to get total flux
print(f"Using aperture correction factor of {apcorr} for radius {radius} pixels and eefraction {eefraction_value}")

# Create apertures for aperture correction, not using the curve of growth
positions = np.transpose((sources['x_centroid'], sources['y_centroid']))

# Redo the aperture photometry using the radius, sky_in, and sky_out from the apcorr file
aperture = CircularAperture(positions, r=radius)
sky_annulus = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)
phot_table_apcorr = aperture_photometry(img_sub, aperture, wcs=wcs_apcorr, method=phot_method)
sky_table_apcorr = aperture_photometry(img_sub, sky_annulus, wcs=wcs_apcorr, method=phot_method)

# Extract flux and sky
fluxes = phot_table_apcorr['aperture_sum'].value  # MJy/sr
sky_mean = sky_table_apcorr['aperture_sum'].value / sky_annulus.area  
sky_total = sky_mean * aperture.area  

# Correct for sky
net_fluxes = fluxes - sky_total  

# Apply aperture correction 
total_fluxes = net_fluxes * apcorr  

# Convert to AB magnitudes using the pixel area in steradians from the header
pixarea_sr = header['PIXAR_SR']  # in steradians
# total_flux_jy = total_fluxes * 1e6  # MJy → Jy
abmag_apcorr = convert_aperture_sum_Jy_per_sr_to_abmag(total_fluxes, header=header)

# Add column to the phot_table_apcorr with the aperture-corrected AB magnitudes
phot_table_apcorr['ABmag_apcorr'] = abmag_apcorr

# Create merged table on the x and y coordinates of the sources to compare 
# the aperture-corrected magnitudes with the original photometry catalog
merged_table = join(phot_table_apcorr, catalog, keys=['xcenter', 'ycenter'])


# Make a histogram of the aperture sums in the catalog
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(catalog['aperture_sum_abmag'], bins=30, alpha=0.5, label='Original photometry')
ax.hist(phot_table_apcorr['ABmag_apcorr'], bins=30, alpha=0.5, label='Aperture-corrected')
ax.set_xlabel('Aperture Sum')
ax.set_ylabel('Frequency')
ax.legend()
plt.show()


