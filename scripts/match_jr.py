from astropy.table import Table, hstack
import os
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

pl.ion()

os.chdir("/Users/ri3e/cv/galaxies/phot/")

jr=Table.read("ngc3351_pop2plus3_filtered.csv", format="ascii.csv")
ri=Table.read("from_33pah/ngc3351_combined.ecsv")

with open("ngc3351_pop2plus3_filtered.reg", "w") as f:
    f.write("fk5\n")  # Assuming the region file requires a coordinate system header
    for row in jr:
        f.write(f"circle({row['raj2000']},{row['dej2000']},0.1\")\n")


def match(
    catalog0, 
    catalog1, 
    rmax=1, # arcsec
    keys=[['raj2000','ra'], ['dej2000', 'dec']]):

    coords0 = SkyCoord(catalog0[keys[0][0]], catalog0[keys[1][0]], unit=u.deg)
    coords1 = SkyCoord(catalog1[keys[0][1]], catalog1[keys[1][1]], unit=u.deg)
    indices, separations, _ = match_coordinates_sky(coords0, coords1)
    match_mask = separations <= rmax * u.arcsec

    # Build matched catalog
    matched0 = catalog0[match_mask]
    matched1 = catalog1[indices[match_mask]]

    unmatched = catalog0[~match_mask]
    #print(unmatched)

    # Optionally combine columns from both catalogs
    matched_cat = hstack([matched0, matched1])
    return matched_cat, separations[match_mask].degree

center_on_JR = False
if center_on_JR:
    # this finds the nearest PAH source for each JR excess source
    matched_cat, distances = match(jr, ri, keys=[['raj2000','ra'], ['dej2000', 'dec']], rmax=0.5)
else:
    # this finds the nearest JR excess source for each PAH source:
    matched_cat, distances = match(ri, jr, keys=[['ra','raj2000'], ['dec', 'dej2000']],rmax=0.5)

distances *= 3600

zord=np.argsort(distances)[::-1]
unsort = np.arange(len(matched_cat))[zord] # get back to original row
matched_cat = matched_cat[zord]
distances = distances[zord]


pl.clf()
pl.plot((matched_cat['ra']-matched_cat['raj2000'])*3600, (matched_cat['dec']-matched_cat['dej2000'])*3600, 'o')
pl.xlabel('RA difference (arcsec)')
pl.ylabel('Dec difference (arcsec)')


z=np.where(distances>0.15)[0]
for iz in z:
    print(f"Match {iz}: distance = {distances[iz]}")

jr_full = Table.read("Phot_v5p4_src_335_ngc3351.csv") #Phot_v5p3_src_335_ci200ngc3351.csv")

# we match based on "ra""dec" which is the pahsub position, i.e. we're lookig for the nearest JR source to the pah-sub position
match2, distances2 = match(matched_cat,jr_full, keys=[['ra','raj2000'], ['dec','dej2000']], rmax=0.5)
distances2 *= 3600

# after the match, raj2000 will become raj2000_1 from the matched_cat i.e. jimena excess source position
# raj2000_2 will be from the full jimena list

pl.plot((match2['ra']-match2['raj2000_2'])[z]*3600, (match2['dec']-match2['dej2000_2'])[z]*3600, 'o')

pl.clf()
pl.plot(distances,distances2, 'o')
pl.xlabel('PAH peak to JR excess position (arcsec)')
pl.ylabel('PAH peak to jr_full (arcsec)')

pl.clf()
pl.plot(jr_full['F300M_ab']-jr_full['F335M_ab'],jr_full['F335M_ab'],'k.',alpha=0.2,label='All JR sources',markersize=2)
pl.plot(matched_cat['F300M_ab']-matched_cat['F335M_ab'],matched_cat['F335M_ab'],'.',label='JR excess sources')
z=np.where(distances2<distances)[0]
pl.plot(match2['F300M_ab_2'][z]-match2['F335M_ab_2'][z],match2['F335M_ab_2'][z],'.',label='closer JR sources')

pl.xlabel('F300M_ab - F335M_ab')
pl.ylabel('F335M_ab')
pl.ylim(23,18)
pl.xlim(-1,5)
pl.legend(prop={'size': 8})
if center_on_JR:
    pl.savefig("CMD_JR_with_no_PAHsource.png")
else:
    pl.savefig("CMD_PAH_with_no_JRsource.png")

hdupah = fits.open("../phangsdata/ngc3351/v2p0_ngc3351_convolved_pah_f360m.fits")
# hdu335 = fits.open("../phangsdata/ngc3351/ngc3351_nircam_lv3_f335m_i2d_anchor.fits")['SCI']
hdu335 = fits.open("../phangsdata/ngc3351/ngc3351_nircam_lv3_f300m_i2d_anchor.fits")['SCI']


def imclip(hdu,ra,dec,size):
    w = WCS(hdu.header)
    x, y = w.all_world2pix(ra, dec, 0)
    x = int(x)
    y = int(y)
    half_size = size // 2
    xyoff = (x-half_size, y-half_size)
    return hdu.data[y-half_size:y+half_size, x-half_size:x+half_size], xyoff


pl.clf()
pl.subplots(3,4)
z=np.arange(12) # first 12 

w335=WCS(hdu335.header)

all_pop3 = w335.all_world2pix(jr['raj2000'], jr['dej2000'], 0)
all_full = w335.all_world2pix(jr_full['raj2000'], jr_full['dej2000'], 0)
all_pah = w335.all_world2pix(ri['ra'], ri['dec'], 0)

for i in range(12):
    pl.subplot(3,4,i+1)
    if center_on_JR:
        im, xyoff = imclip(hdu335, matched_cat['raj2000'][z[i]], matched_cat['dej2000'][z[i]], 20)
    else:
        im, xyoff = imclip(hdu335, matched_cat['ra'][z[i]], matched_cat['dec'][z[i]], 20)

    pl.imshow(np.log10(im), origin='lower', cmap='Grays')
    pl.xticks([])
    pl.yticks([])

    for j in range(len(all_pop3[0])):
        if all_pop3[0][j] > xyoff[0] and all_pop3[0][j] < xyoff[0]+20 and all_pop3[1][j] > xyoff[1] and all_pop3[1][j] < xyoff[1]+20:
            pl.plot(all_pop3[0][j]-xyoff[0], all_pop3[1][j]-xyoff[1], 'bs', markersize=6, mfc="none")
    for j in range(len(all_full[0])):
        if all_full[0][j] > xyoff[0] and all_full[0][j] < xyoff[0]+20 and all_full[1][j] > xyoff[1] and all_full[1][j] < xyoff[1]+20:
            pl.plot(all_full[0][j]-xyoff[0], all_full[1][j]-xyoff[1], 'x', color="orange",markersize=5)
    for j in range(len(all_pah[0])):
        if all_pah[0][j] > xyoff[0] and all_pah[0][j] < xyoff[0]+20 and all_pah[1][j] > xyoff[1] and all_pah[1][j] < xyoff[1]+20:
            pl.plot(all_pah[0][j]-xyoff[0], all_pah[1][j]-xyoff[1], 'ro' ,markersize=5, mfc="none")
    
    xy=w335.all_world2pix(matched_cat['ra'][z[i]], matched_cat['dec'][z[i]], 0)
    pl.plot(xy[0]-xyoff[0], xy[1]-xyoff[1], 'ro', markersize=10, mfc="none")
    xy=w335.all_world2pix(matched_cat['raj2000'][z[i]], matched_cat['dej2000'][z[i]], 0)
    pl.plot(xy[0]-xyoff[0], xy[1]-xyoff[1], 'cs', markersize=8, mfc="none")
    xy=w335.all_world2pix(match2['raj2000_2'][z[i]], match2['dej2000_2'][z[i]], 0)
    pl.plot(xy[0]-xyoff[0], xy[1]-xyoff[1], '+', color="orange",markersize=8)

    pl.text(0.05, 0.95, f"pop3 ID: {matched_cat['ID_source'][z[i]]}", transform=pl.gca().transAxes, color='black', fontsize=8, verticalalignment='top')
    pl.text(0.05, 0.87, f"pop3 row: {unsort[z[i]]+1}", transform=pl.gca().transAxes, color='black', fontsize=8, verticalalignment='top')
    
pl.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95,left=0.05,bottom=0.05,top=0.95)

if center_on_JR:
    pl.savefig("cutouts_JR_with_no_PAHsource.png")
else:
    pl.savefig("cutouts_PAH_with_no_JRsource.png")

# large distances according to janice: rows 29,23,9,21,38,34,17,32

z=np.where((distances > 0.2)*(distances2>0.1))[0]  

with open("ngc3351_pah_no_JRsource.reg", "w") as f:
    f.write("fk5\n")  
    for i in z:
        f.write(f"circle({matched_cat['ra'][i]},{matched_cat['dec'][i]},0.1\")\n")


print("buh")