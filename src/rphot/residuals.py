import math
import warnings
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.table import QTable
from astropy.visualization import ImageNormalize, LogStretch
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import rotate

from .io import load_filter_data, get_psf_file
from .units import im2flux, flux2im

filter_fwhm_pix, filter_wave = load_filter_data()

# Take an image clip and return the residual after subtracting the PSFcores of sources as defined by params
def residual(params,imclip,return_type,psfcore,scor_pix,sfitrgn_pix,njparams):
    # params = [fluxes,jtypes] # jtypes could be fixed
    # the "f" parameters are the fluxes of the main and neighboring sources
    fparms=[x for x in params.keys() if x[0]=='f']

    nsrc=len(fparms) 
    outclip = imclip.copy() 
    if return_type == "res4fit" or return_type == "suball":
        outclip -= params['bg'] 
    mask=np.ones(outclip.shape,dtype="bool")
        
    for i in range(nsrc):
        dj,j=math.modf(params['j%i'%i])
        # x1,y1 parameter is the offset in full pixel units
        off=np.array([params['x%i'%i],params['y%i'%i]])
        off0=np.int32(off) # off0 is nearest full pixel  
  
        dx,x=math.modf(4*off[0]-4*off0[0]) # x,dx are in 1/4-pix units, and relative to the nearest whole pixel off0
        dy,y=math.modf(4*off[1]-4*off0[1])
        x=int(x)+2 # extra +2 because psf is centered between whole pixels
        y=int(y)+2
        j=int(j)
        
        # find broadened psfs bracketing the parameter j1 (and interpolate below)
        j1=j+1
        if j1>=njparams-1:
            j1=njparams-1

        # these psfs are also binned back to the original pixel scale from 4x
        q11 = np.roll(psfcore[j ],[x  ,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
        q21 = np.roll(psfcore[j ],[x+1,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
        q12 = np.roll(psfcore[j ],[x  ,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
        q22 = np.roll(psfcore[j ],[x+1,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
        j0model = q11*(1-dx)*(1-dy) +q12*(1-dx)*dy +q21*dx*(1-dy) +q22*dx*dy            
  
  
        # turn off qq calculation if njparams=1 i.e. no broadening
        if njparams>1:
            r11 = np.roll(psfcore[j1],[x  ,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
            r21 = np.roll(psfcore[j1],[x+1,y  ],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
            r12 = np.roll(psfcore[j1],[x  ,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
            r22 = np.roll(psfcore[j1],[x+1,y+1],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
            j1model = r11*(1-dx)*(1-dy) +r12*(1-dx)*dy +r21*dx*(1-dy) +r22*dx*dy
                                      
        if params['j%i'%i]>0:
            if j<njparams-1:
                imodel=j0model*(1-dj) + j1model*dj
            else:
                imodel=j1model
        else:
            imodel=j0model
  
        # now we need to full-pixel location of the imodel (psf core size scor_pix) inside the image clip
        brd=sfitrgn_pix//2-scor_pix//2 # border between clip and psfcore
        # where to place in x, if not off edge
        xou=[sfitrgn_pix[1]//2+off0[1]-scor_pix[1]//2, sfitrgn_pix[1]//2+off0[1]+scor_pix[1]//2]
        xin=[0,scor_pix[1]+1] # index in psfcore
        you=[sfitrgn_pix[0]//2+off0[0]-scor_pix[0]//2, sfitrgn_pix[0]//2+off0[0]+scor_pix[0]//2]
        yin=[0,scor_pix[0]+1] # index in psfcore
        if off0[1]<-brd[1]: # off left side
            xou[0]=0
            xin[0]=-off0[1]-brd[1]
        if off0[1]>=brd[1]: # off right side
            xou[1]=sfitrgn_pix[1]
            xin[1]=scor_pix[1]-(off0[1]-brd[1])
        if off0[0]<-brd[0]: # off bottom
            you[0]=0
            yin[0]=-off0[0]-brd[0]
        if off0[0]>=brd[0]: # off top
            you[1]=sfitrgn_pix[0]
            yin[1]=scor_pix[0]-(off0[0]-brd[0])
  
        if params['f%i'%i].vary==True:  # 2026 TODO make sure this is what we want - to not subtract unfit neighbors
            outclip[xou[0]:xou[1],you[0]:you[1]] -= \
                params['f%i'%i]*imodel[xin[0]:xin[1],yin[0]:yin[1]]
            mask[xou[0]:xou[1],you[0]:you[1]] = False
            # TODO mask tighter for psf peak only?
 
    if return_type=="res4fit":
        # TEST SOFTENING of residual - for f1000, SQRT helps, 1/3 doesn't make much difference.
        #z=np.where(outclip>0)
        #outclip[z]=(outclip[z])**(1/4)
        #outclip[z]=np.sqrt(outclip[z])

        # z=np.where(outclip<0)
        # outclip[z]=-(-outclip[z])**(1/4)
        #outclip[z]=-np.sqrt(-outclip[z])

        # this masks parts of the clip w/o any fitted sources - required?
        # outclip[np.where(mask)]=0
        return outclip # TODO - weight small scales by subtracting the median?
    else:
        return outclip




def fit_and_subtract(infile, # input mosaic image 
                gal=None,
                band="pah33",
                srcfile=None, # source catalog to use for fitting 
                pixbinfactor=1.,
                fittype="amp", # "amp" or "amppos" or "ampwid" or None to just create residual
                doplot=True,
                kflux='flux_F1000W', # key name for app flux in input catalog
                kdflux='fluxerr_F1000W',  # key name for app flux error in input catalog
                kra='raj2000', # key name for RA in input catalog
                kde='dej2000', # key name for Dec in input catalog
                doregion=False,
                rd0=[0,0], # center of region to fit in degrees
                d=0.01, # size of region to fit in degrees
                radius=None,
                # TODO make flux limits unit-aware
                lowfluxlim = 1e-5, # below this its a nondetection           
                psf_dir="./",
                cat_filetype="ecsv",
                local={}, config={}
                ):

    from lmfit import minimize, Parameters, create_params


    maxfluxfactor=5 # limit on how much brighter the source can get when fit
    
    # rotational angle miri is V3 +~5
    # PA_APER =   245.97609259623073 / [deg] Position angle of aperture used
    # PA_V3   =   241.05116909076034 / [deg] Position angle of telescope V3 axis

    imfloor=1 # add a floor to the image for display only
    origfluxcut = 0.5 # filter for only sources with aperture flux above this value
    fiteverything=True # if False, only fit crowded sources
    plotborder=0 # fraction of sfitrgn_pix to include around the border when displaying the region
    alpha=0.5 # for overplotting sources
    larger_sfitrgn_pix=False
    debug=False # stop every source and show neighbors etc
    file_root = f"{gal}_{band}"
    
    #---------------------------------------------
    # load input image
    # TODO use load_jwst instead of manually opening the FITS file and handling extensions
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
        for whathdu in ['SCI','CONTSUB','PRIMARY']:
            if whathdu in fits.open(infile):
                break
        inhdu = fits.open(infile)[whathdu]
        header = inhdu.header
        inwcs = wcs.WCS(header)

    # pixel scale in arcsec/pixel
    pixsize = wcs.utils.proj_plane_pixel_scales(inwcs) * 3600

    # if this is a simulated image moved to a larger distance, then the pixels are artificially large
    # because the simulated images have large pixels in the WCS, so that the coordinates of sources
    # match the original nearby image. If you're not doing simulated images, pixbinfactor should be 1.
    pixsize /= pixbinfactor
    
    # Sr per pixel, used to convert between mJy flux density and MJy/Sr peak value
    srperpix=(pixsize[0]/206265)**2

    fwhm_pix=filter_fwhm_pix[band.upper()]
    wave=filter_wave[band.upper()]

    
    #---------------------------------------------
    # read in input source list and image
    srclist=QTable.read(srcfile)
    # in fits format this comes in as a masked array which breaks other things later
    if np.ma.is_masked(srclist[kflux]):
        srclist[kflux] = srclist[kflux].filled(0)
    
    # order from brightest to faintest
    # TODO refactor to do this only in the fit loop, to not disorder the actual list
    # and keep it identical to the input order
    sortidx=np.argsort(srclist[kflux])[::-1]
    
    nsrc=len(srclist)
    srcra=srclist[kra].data
    srcde=srclist[kde].data
    
    #---------------------------------------------
    # set up psfs - expects 4 x oversampled psf in the file
    psffile = get_psf_file(band,psf_dir=psf_dir)

    psfdat=fits.getdata(psffile)
    #OVERSAMP=                    4 / Oversampling factor for FFTs in computation
    #DET_SAMP=                    4 / Oversampling factor for MFT to detector plane
    #PIXELSCL=               0.0277 / Scale in arcsec/pix (after oversampling)
    
    # make psfs broadened by "widths" QUARTER-pixel amounts:
    # widths=np.int32(np.round(np.array([0,4,8])*wave/21))+1 # scale by wavelength
    # doesn't really do much for for miri;
    # 3.6sub has point sources at widths of 0.19" up to compact sources of 0.32"
    # 4 pixel width =16 quarter pix should get us up to the more extended sources
    widths=[0,2,6,10] 
    # finer grid 
    widths=np.arange(10) 
    nbroad=len(widths)
    
    s=psfdat.shape
    # this "p" array will store broadened psfs, at quarter-pix resolution
    p=np.zeros([nbroad,s[0],s[1]]) 
    
    # ROTATE psf to the correct position angle
    if 'pa_aper' in inhdu.header:
        psfdat=rotate(psfdat,-inhdu.header['pa_aper'],reshape=False)
    else:
        print("WARNING: can't rotate PSF because image position angle not found in header.")
    
    for k in range(nbroad):
        p[k]=gf(psfdat,widths[k])
        # p[k]=gf(psfdat,widths[k]).reshape(s[0]//4,4,s[1]//4,4).sum(3).sum(1)
        # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
        p[k]/=p[k].sum()
    
    #---------------------------------------------
    # set up for psf-fitting 
    # the psf fitting will only use the core of the psf, defined here of size
    #   2*npixrcore in quarter-pixels = scor_qpix
    # the subimage over which the fitting is done is sized "sfitrgn_pix" in fullpixels,
    #    or "window" in degrees
    # sources inside rfit radius will be fitted
    # sources inside sfitrgn_pix, but outside of rfit, will be subtracted with their current flux in the flux catalog during the fit
    # then scor_pix is the size of the core psf in full pixels, scor_qpix//4
    
    # keep track of any sources that get their fluxes adjusted by fitting
    fitted=np.zeros(nsrc)
    # and the fitted j parameter (width index)
    jout=np.zeros(nsrc)
    # and the fitted width
    wout=np.zeros(nsrc)
    # and fitted xy position
    xyout=np.zeros([nsrc,2])
    rdout=np.zeros([nsrc,2])
    rdout[:,0]=srclist['ra']
    rdout[:,1]=srclist['dec']
    # distance to the nearest source
    nearest=np.zeros(nsrc)

    # the new fitted flux (will be filled during fitting)
    newflux=srclist[kflux].copy()
    
    # what is distance of proximity that sources have to be fit together?
    # use 2*fwhm
    rfit=fwhm_pix*pixsize[0]*2/3600
    
    # for F2100 I did 70:111 i.e. a half-width of 6.7*FWHM or 2.7*first null
    # npixrcore=int(round(7*fwhm/2/pixsize[0]*4)) # quarter-pixels
    # that seems a bit much 1000
    # npixrcore=int(round(6*fwhm_pix/4))*8 # quarter-pixels
    # actually this mostly depends on how much widening there is

    npixrcore=int(round(fwhm_pix + np.max(widths)/8))*8 # quarter-pixels

    # scor_pix is npixrcore*2, modulo being even - this defines the size of the core psf used for fitting
    
    # this is the half-size of the input psf in quarter-pixels
    spsf=psfdat.shape[0]//2 # this is even - the psf is centered between pixels 
    icore=[spsf-npixrcore,spsf+npixrcore]

    # this is the core region of the psf in quarter-pixels
    psfcore=p[:,icore[0]:icore[1],icore[0]:icore[1]]
    # TODO: divide by sum of psfcore, to get the flux normalization right later when making the residual?
    scor_qpix=np.array(psfcore.shape[-2:])
    # scor_pix is the size of psfcore in full pixels
    scor_pix = scor_qpix//4  # has to be even

    # rpix is used for plotting; it could be different from scor_pix/2 if we wanted to visualize e.g. the first null
    rpix = scor_pix[0]/2
    
    # psfcore is centered at 39.5,39.5 = scor_pix/2-0.5
    degperpix=pixsize[0]/3600
    
    # sfitrgn_pix is the size of the fitting region in full pixels - it has to at least fit the core psf
    # tried 2*scor_pix but that seems a lot... maybe 1.5
    sfitrgn_pix=np.int32(np.round(1.5*scor_pix))
    # the distance in degrees within which we have to find sources to fit along with the source being considered
    # window=(sfitrgn_pix[0]//2)*degperpix # equal to the half-size of the fitting region - find everything
    window=(sfitrgn_pix[0]//2-scor_pix[0]//2)*degperpix # add border
    
    if larger_sfitrgn_pix:
        sfitrgn_pix *=2 
        file_root += "_sfitrgnx2"
    
    if sfitrgn_pix[0]/2!=sfitrgn_pix[0]//2:
        raise ValueError("sfitrgn_pix is odd")
    
    if doregion:
        file_root += "_region"
        cdec=np.cos(rd0[1]*np.pi/180)
        subim_xy=np.int32(inwcs.wcs_world2pix([[rd0[0]-d/cdec,rd0[1]-d],[rd0[0]+d/cdec,rd0[1]+d]],0))
        if larger_sfitrgn_pix: # does this work?  seems off?
            subim_xy+=np.array([[sfitrgn_pix[1]//4,-sfitrgn_pix[0]//4],[-sfitrgn_pix[1]//4,sfitrgn_pix[0]//4]])   
        subim=inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]
    else:
        subim=inhdu.data
        subim_xy=[[subim.shape[1],0],[0,subim.shape[0]]]

    subim *= u.Unit(header['BUNIT'])

    # fitted local background/ floor level
    bgfit=np.zeros(nsrc) * subim.unit

    
    # initialize the residual image for fitting, if there is going to actually be any fitting.
    if fittype is not None:
        resid=subim.copy() 
        model=np.zeros_like(subim)
    # params dx,dy are offsets from the central pixel of the stack for each src
    # with psfdat in hand, we go to closest quarter-pixels in the residual image
         
    # this will store the residual from photometry before fitting i.e. from the app phot
    resid_nofit=subim.copy()
    # this will just be the stars, using the original photometry
    model_nofit=np.zeros_like(subim)
    
    
    
    #---------------------------------------------
    # subplot 0 is the original image, zoomed if doregion=True
    if doplot:          
        fig = plt.figure(0,figsize=(8,8))
        plt.clf()
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.1,right=0.98)
        plt.subplot(2,2,1)
        norm = ImageNormalize(vmin=np.nanpercentile(subim.value, 10),
                              vmax=np.nanpercentile(subim.value, 99.99),
                              stretch=LogStretch())
        cmap4ims = "viridis"
        plt.imshow(subim.value, origin='lower', cmap=cmap4ims, norm=norm)
        if doregion:
            ax=plt.gca()
            #ax.use_sticky_edges=False
            #ax.margins(x=-0.4,y=-0.4)
            # zoom because of border
            s=subim.shape
            ax.set_xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
            ax.set_ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
        
        plt.title("original image")
        plt.xticks([])

              
    # open a ds9 output file
    #if fittype:
    #     ds9reg=open('.'.join(srcfile.split('.')[:-1])+"_"+fittype+".reg","w")
    #     ds9reg.write("fk5\n")

    th=np.arange(21)/10*np.pi
    st=np.sin(th)
    ct=np.cos(th)
    

    #---------------------------------------------
    # the loop over sources
    for i0 in range(nsrc):

        i=sortidx[i0] # sorted order bright->faint
        # TODO make this unit-aware
        if (srclist[kflux][i].value < lowfluxlim):
            continue

        if i0//500==i0/500:
            print(f"Processing source {i0} (sorted index {i})") # pixel position of the source in the subimage
        xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
        
        # only deal with sources where the entire regular-pix-size psfcore (sfitrgn_pix) fits within the subim
        # (the subim is the entire image if doregion=False)
        if srclist[kflux][i]>0 and xy.min()>(sfitrgn_pix[0]/2) and xy[1]<(subim.shape[0]-sfitrgn_pix[0]//2) and xy[0]<(subim.shape[1]-sfitrgn_pix[1]//2):
     
            xy0=np.int32(np.round(xy))
            # from here to the end of processing this source, everything is relative to xy0,
            # the rounded source position in the original (sub)image coordinates, 
            # which maps to sfitrgn_pix//2 in the small clipped region to be fit
      
            # fractional pixel offset of the source
            off=xy-xy0
            
            # determine nearest 1/4 pixel offset to roll the psf        
            # +2 is b/c psf is centered between whole pixels - verified 20250209
            xy4i=np.int32(np.round(off*4))+2
            # subtract the original aperture photom, unbroadened
            # TODO consider subtracting entire psf, not just core of size scor_pix
            resid_nofit[xy0[1]-scor_pix[1]//2:xy0[1]+scor_pix[1]//2,xy0[0]-scor_pix[0]//2:xy0[0]+scor_pix[0]//2]-= \
                flux2im( srclist[kflux][i], header ) * \
                np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
            model_nofit[xy0[1]-scor_pix[1]//2:xy0[1]+scor_pix[1]//2,xy0[0]-scor_pix[0]//2:xy0[0]+scor_pix[0]//2]+= \
                flux2im( srclist[kflux][i], header ) * \
                np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)

            # if this source has not yet been fitted
            # (if fittype==None, then we'll never go in this block)
            if fittype and fitted[i]<=0:
                # find neighboring sources, within "window" (usually set to sfitrgn_pix*pixsize/2 above)
                cdec=np.cos(srcde[i]*np.pi/180)
                # TODO make flux limit unit-aware
                znear=np.where( (np.absolute(srcra-srcra[i])<1.*window/cdec )*
                                (np.absolute(srcde-srcde[i])<1.*window )*
                                (newflux.value > lowfluxlim))[0]
      
                # if newflux (which is still the input flux) of the main source is bright enough, 
                # and there are some neighbours, they may be part of the Airy ring and should be removed
                if newflux.value[i] > 0.15 and len(znear) > 1 and 'f1000' in band.lower():
                    d2=(srcra[znear]-srcra[i])**2*cdec**2 + (srcde[znear]-srcde[i])**2
                    zring=znear[np.where((d2>(1.1/3600)**2)*(d2<(1.5/3600)**2))[0]] 
                    if len(zring)>0:
                        fitted[zring]=100
                        newflux.value[zring] = lowfluxlim
                        # TODO plot these with different symbol?
                    znear=np.where( (np.absolute(srcra-srcra[i])<1.*window/cdec )*
                                    (np.absolute(srcde-srcde[i])<1.*window )*
                                    (newflux.value > lowfluxlim))[0]
      
                if len(znear)>1: # sort neighbour sources by distance from the primary source
                    d2=(srcra[znear]-srcra[i])**2*cdec**2 + (srcde[znear]-srcde[i])**2
                    znear_ord=znear[d2.argsort()]
                    d2.sort()

                    # distinguish between "very" close - within 1/4 of the fitting radius, and just neighboring 
                    if d2[1] < (rfit**2 / 16):  # /4 or /9= TEST ALLOW CLOSER TO BE WIDENED
                        someclose=True  # some are very close
                    else:
                        someclose=False
                else: # znear includes the source itself, so if it is length==1, its *only* the primary source
                    znear_ord=znear
                    someclose=False
                    d2=np.array([0])

                # fiteverything means to fit all neighbors in the fitregion, 
                # otherwise only fit the primary source and the "very close"
                if fiteverything or someclose:
                    if "wid" in fittype and not someclose: # extra parameters - only do widening if not crowded.
                        # print("widening")
                        njparams=nbroad
                    else:
                        njparams=1
      
                    # NOTE: sfitrgn_pix must be even #pix (20250210 TODO check if still true)
                    # finally, we actually extract the "clip" fitting region:
                    # assume that all unit issues have been resolved upstream, and go unitless here
                    imclip=resid[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                                 xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2].copy().value

                    # for already-subtracted images, also assume that very negative values are not good:
                    # TODO make unit-aware
                    imclip[imclip<-1]=-1

                    # the fitting will deal with the source peak flux in image units (e.g. MJy/sr), 
                    # because its easier to scale the model PSF that way, and later we can 
                    # convert that back to mJy  
                    thisfluxMJy=flux2im(newflux[i], header).value # mJy -> Mjy/sr assuming normalized psf
        
                    params = Parameters()
                    params.add('bg', value=np.nanmean(imclip), min=0)

                    # allow the fitted flux to be between 0 and the maximum flux factor times the initial flux
                    params.add('f0', value=thisfluxMJy,min=0,max=thisfluxMJy*maxfluxfactor)
        
                    # if we're allowing the PSF to widen, we let the "j0" parameter vary - 
                    # j0 is the index in the list of PSF broadening widths, and array of psfcores prepared above
                    if njparams<=1:
                        params.add('j0',value=0,vary=False)
                    else:            
                        params.add('j0',value=0,min=0,max=njparams-1)
        
                    # this will be the fitted offset from xy0 in whole pixel units, 
                    # and is allowed to vary by +/- 1 pixel in both x and y directions
                    params.add('x0',value=off[0],min=off[0]-1,max=off[0]+1)  
                    params.add('y0',value=off[1],min=off[1]-1,max=off[1]+1)

                    # if the position is not being fitted, we fix those offset values x0 and y0
                    if "pos" not in fittype:
                        params['x0'].vary=False
                        params['y0'].vary=False

                    # now go through the neighbors
                    for ii in np.arange(1,len(znear_ord)):
                        
                        # other sources positions, are offset relative to the main source
                        xy_near=inwcs.wcs_world2pix([[srcra[znear_ord[ii]],srcde[znear_ord[ii]]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
                        off=xy_near-xy0

                        # f1 f2 etc are the (peak) fluxes of the neighboring sources in MJy/sr
                        thisfluxMJy=flux2im(newflux[znear_ord[ii]], header).value # original or fitted flux
                        params.add('f%i'%ii, value=thisfluxMJy,min=0,max=thisfluxMJy*maxfluxfactor)
        
                        # don't re-fit previously fitted sources, nor ones outside the rfit fitting radius
                        if d2[ii]>rfit**2 or fitted[znear_ord[ii]]>0:
                            params['f%i'%ii].vary=False
        
                        if njparams<=1:
                            params.add('j%i'%ii,value=0,vary=False)
                        else:
                            params.add('j%i'%ii,value=0,min=0,max=njparams-1)
                        
                        params.add('x%i'%ii,value=off[0],min=off[0]-1,max=off[0]+1)
                        params.add('y%i'%ii,value=off[1],min=off[1]-1,max=off[1]+1)
        
                        if "pos" not in fittype:
                            params['x%i'%ii].vary=False
                            params['y%i'%ii].vary=False
        
                    # TODO put upper bound on bg level
        
                    # lmfit can't deal with nans
                    z=np.where(np.isnan(imclip))
                    # srcstack[0][z]=0
                    # but the nans are on the edge so just don't fit sources with NaNs nearby
                    if len(z[0])>0:
        
                        # allow to get within scor_pix of NaNs (generally only off the edge)
                        s=imclip.shape
                        zctr=np.where(np.isnan(imclip[s[1]//2-scor_pix[1]//2:s[1]//2+scor_pix[1]//2,s[0]//2-scor_pix[0]//2:s[0]//2+scor_pix[0]//2]))[0]
                        if len(zctr)>0:
                            print("NaN too near position")
                            continue
                        else:
                            # if there is a stray NaN in the fitting region, set it to zero
                            imclip[z]=0
        
                    # https://lmfit.github.io/lmfit-py/
                    out=minimize(residual,params, args=[imclip,"res4fit",psfcore,scor_pix,sfitrgn_pix,njparams],nan_policy="omit")

                    if out.success:
                        # TODO this may be ~10% high because we fit the PSF core vs full PSF
                        newflux[i] = im2flux( out.params['f0'] * subim.unit, header )
                        fitted[i]=len(np.where(d2<rfit**2)[0]) # number of sources fitted TODO this doesn't take into account if some near sources have not be re-fit because they were previously fit
                        jout[i] =out.params['j0']
                        wout[i] =np.interp(out.params['j0'].value, range(len(widths)), widths)
                        xyout[i]=np.array([out.params['x0'].value,out.params['y0'].value])+xy0
                        bgfit[i]=out.params['bg'] * subim.unit
                        rdout[i]=inwcs.wcs_pix2world([xyout[i]],0)[0]
        
                        # set nearby *fitted* values and jout, xyout,
                        # so they don't get fit again later
                        for ii in np.arange(1,len(znear_ord)):
                            xynear=np.array([out.params['x%i'%ii].value,out.params['y%i'%ii].value])+xy0
                            xyout[znear_ord[ii]]=xynear
                            rdout[znear_ord[ii]]=inwcs.wcs_pix2world([xynear],0)[0]
                            if out.params['f%i'%ii].vary==True:
                                fitted[znear_ord[ii]]=fitted[i]
                                jout[znear_ord[ii]]=out.params['j%i'%ii]
                                wout[znear_ord[ii]] =np.interp(out.params['j%i'%ii].value, range(len(widths)), widths)
                                bgfit[znear_ord[ii]]=out.params['bg'] * subim.unit
                                # subtract the fitted neighbor from the original-photometry residual image
                                # we do this here because we won't consider a neighbor that gets fitted again later
                                # if we're doing actual fitting.
                                # If we're not doing actual fitting we'll never get here, 
                                # so this subtraction only happens during actual fitting.
                                xy0near=np.int32(np.round(xynear))                                        
                                xy4i=np.int32(np.round((xynear-xy0near)*4))
                                resid_nofit[xy0near[1]-scor_pix[1]//2:xy0near[1]+scor_pix[1]//2,xy0near[0]-scor_pix[0]//2:xy0near[0]+scor_pix[0]//2]-= \
                                    flux2im( srclist[kflux][znear_ord[ii]], header ) * \
                                    np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
                                
                                model_nofit[xy0near[1]-scor_pix[1]//2:xy0near[1]+scor_pix[1]//2,xy0near[0]-scor_pix[0]//2:xy0near[0]+scor_pix[0]//2]+= \
                                    flux2im( srclist[kflux][znear_ord[ii]], header ) * \
                                    np.roll(psfcore[0],[xy4i[0],xy4i[1]],axis=[1,0]).reshape(scor_pix[0],4,scor_pix[1],4).sum(3).sum(1)
                                          
                        # if this source has been successfully fit,
                        # replace its params with the new fitted ones 
                        params=out.params 
        
                    else: # failed fit
                        fitted[i]=-1
                        
                    if len(znear)>1:
                        nearest[i]=np.sqrt(d2[1])
      
                # calling residual(return_type=substars) will only remove the fitted sources and 
                # not remove a flat bg i.e. its designed for creating the residual image
                # return_type = suball will also subtract the constant background
                thisresid = residual(params,imclip,"substars",psfcore,scor_pix,sfitrgn_pix,njparams) * subim.unit
                # TODO subtract entire psf, not just core, here (would need the residual function to do that functionality, so probably not worth it
                thismodel = (imclip + params['bg'])*thisresid.unit - thisresid
      
                resid[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                    xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2] = thisresid

                model[xy0[1]-sfitrgn_pix[1]//2:xy0[1]+sfitrgn_pix[1]//2,
                    xy0[0]-sfitrgn_pix[0]//2:xy0[0]+sfitrgn_pix[0]//2] = thismodel

                if debug: # or (newflux[i]/srclist[kflux][i] > 2.5):
                    plt.clf()
                    plt.imshow(np.concatenate([imclip,thisresid.value],axis=1),cmap="jet",origin="lower")
                    plt.title("Source %d: New flux = %.3f, Old flux = %.3f, bg = %.2f, wid = %.1f" % 
                            (i, newflux[i].value, srclist[kflux][i].value, params['bg'].value, jout[i]))
                    for ii in np.arange(0,len(znear_ord)):
                        xynear=np.array([out.params['x%i'%ii].value,out.params['y%i'%ii].value]) + sfitrgn_pix[1]//2
                        if ii==0:
                            if fitted[0]>0:
                                plt.plot(xynear[0]+ct*rpix,xynear[1]+st*rpix,'k',alpha=alpha,linewidth=2)
                            else:
                                plt.plot(xynear[0]+ct*rpix,xynear[1]+st*rpix,'r',alpha=alpha,linewidth=2)
                        elif out.params['f%i'%ii].vary==True:
                            # plot the neighbor as magenta if it has been fitted
                            plt.plot(xynear[0]+ct*rpix/2,xynear[1]+st*rpix/2,'m',alpha=alpha,linewidth=2)
                        else:
                            # and cyan otherwise
                            plt.plot(xynear[0]+ct*rpix/2,xynear[1]+st*rpix/2,'c',alpha=alpha,linewidth=2)
                    continue # PUT BREAKPOINT HERE

            # / end of "if actually fitting this source" block
     
        else: # this source is outside of the fittable region of the (sub)image
            fitted[i]=-2
                      
        # write all sources to ds9 
        #if fittype: 
        #     ds9reg.write("point(%f,%f) # point=circle color=green\n"%(rdout[i][0],rdout[i][1]))
            
    #if fittype:
    #     ds9reg.close()
    
    
    
    # =============== display the original/apphot residual image
    if doplot:
        plt.subplot(2,2,2)
        plt.imshow(resid_nofit.value,norm=norm,cmap=cmap4ims,origin="lower")
        if doregion:
            s=subim.shape
            plt.xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
            plt.ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
            
        plt.title("apphot residual")
        plt.xticks([])     
    

    if fittype is not None:
        if doplot:
            # ========================================
            # now show the residual after psf-fitting 
            plt.subplot(2,2,3)
            plt.imshow(resid.value, origin='lower', cmap=cmap4ims, norm=norm)
            plt.title(f'{fittype} resid (rfit={rpix*pixsize[0]:.2f}")')
            plt.xticks([])
             
        # save the residual image after fitting
        if doregion:
            s=subim.shape
            plt.xlim(plotborder*sfitrgn_pix[0],s[1]-plotborder*sfitrgn_pix[0])
            plt.ylim(plotborder*sfitrgn_pix[1],s[0]-plotborder*sfitrgn_pix[1])
            inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=resid
            inhdu.writeto(local['out_image_dir'] + "/" + file_root+"_resid_region_"+fittype+".fits",overwrite=True)
        else:
            inhdu.data=resid.value
            inhdu.writeto(local['out_image_dir'] + "/" + file_root+"_resid_"+fittype+".fits",overwrite=True)

        # save the model image after fitting
        if doregion:
            inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=model.value
            inhdu.writeto(local['out_image_dir'] + "/" + file_root+"_model_region_"+fittype+".fits",overwrite=True)
        else:
            inhdu.data=model.value
            inhdu.writeto(local['out_image_dir'] + "/" + file_root+"_model_"+fittype+".fits",overwrite=True)


    # save the residual image without fitting
    resid_suffix = f"_resid_apphot_r{radius:4.2f}" if radius is not None else "_resid_apphot"
    if doregion:
        inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=resid_nofit.value
        inhdu.writeto(local['out_image_dir'] + "/" + file_root+resid_suffix+"_region.fits",overwrite=True)
    else:
        inhdu.data=resid_nofit.value
        inhdu.writeto(local['out_image_dir'] + "/" + file_root+resid_suffix+".fits",overwrite=True)
    # save the model image without fitting
    model_suffix = f"_model_apphot_r{radius:4.2f}" if radius is not None else "_model_apphot"
    if doregion:
        inhdu.data[subim_xy[0][1]:subim_xy[1][1]+1,subim_xy[1][0]:subim_xy[0][0]+1]=model_nofit.value
        inhdu.writeto(local['out_image_dir'] + "/" + file_root+model_suffix+"_region.fits",overwrite=True)
    else:
        inhdu.data=model_nofit.value
        inhdu.writeto(local['out_image_dir'] + "/" + file_root+model_suffix+".fits",overwrite=True)


    if fittype is not None:
        srclist['psffit_ra'] = rdout[:,0]
        srclist['psffit_dec'] = rdout[:,1]
        srclist.add_columns([newflux,jout,wout,fitted,bgfit],
                            names=["psffit_"+fittype+'_flux',
                                   'psffit_'+fittype+'_jout',
                                   'psffit_'+fittype+'_wout',
                                   'psffit_'+fittype+'_nfitted',
                                   'psffit_'+fittype+'_bkg'])
        source_file = file_root + "_phot_cat"
        if 'phot_cat_extrastring' in config['parameters']['photometry']:
            if config['parameters']['photometry']['phot_cat_extrastring'] is not None and len(config['parameters']['photometry']['phot_cat_extrastring']) > 0:
                source_file += "_" + config['parameters']['photometry']['phot_cat_extrastring']
        srclist.write(local['out_dir'] + "/" + source_file+"."+cat_filetype,overwrite=True)
        panels=[1,2,3]
    else:
        panels=[1,2]
    
    
    if doplot:
        # standard overplotting
        for i in range(nsrc):

            xy=inwcs.wcs_world2pix([[srcra[i],srcde[i]]],0)[0]-np.array([subim_xy[1][0],subim_xy[0][1]])
            if fitted[i]<0:

                if fitted[i]==-2:
                    psym='s'
                    col='k'
                    ms=5
                    alpha=0.5
                elif fitted[i]==-1: # not fit
                    psym='x'
                    col='k'
                    ms=5
                    alpha=1

                for jj in panels:
                    plt.subplot(2,2,jj)
                    plt.plot(xy[0],xy[1],psym,color=col,alpha=alpha,markersize=ms,linewidth=1,mfc="none")

            else: 
                wid=1
                ms=3
                if newflux[i].value > lowfluxlim:
                    fratio=newflux[i]/srclist[kflux][i]
                    alpha=1
                    ms=3
                    r=rpix
                    if fratio<0.1:
                        col='r'
                        alpha=0.5
                    elif fratio<0.5:
                        col='m'
                    elif fratio>3:
                        col='c'
                    else:
                        col='k'
                        alpha=0.5
                else:
                    col='y'
                    alpha=0.5
                    r=rpix/2

                if fittype:
                    plt.subplot(2,2,4)
                    if newflux[i].value < lowfluxlim:
                        plt.plot(srclist[kflux][i].value,lowfluxlim,'v',color=col,ms=1+wout[i],mfc="none")
                    else:
                        plt.plot(srclist[kflux][i].value,newflux[i].value,'o',color=col,ms=wout[i],mfc="none")
                for jj in panels:
                    plt.subplot(2,2,jj)
                    plt.plot(xy[0]+ct*r,xy[1]+st*r,color=col,alpha=alpha,markersize=ms,linewidth=wid,mfc="none")
         
                     
    
        if fittype:
            plt.subplot(2,2,4)
            plt.plot(plt.xlim(),plt.xlim(),'k',alpha=0.2)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("aperture photometry")
            plt.ylabel("psf-fitted photometry")
            plt.savefig(local['out_dir'] + "/" + file_root+"_"+fittype+"_residuals.png")
        else:
            plt.savefig(local['out_dir'] + "/" + file_root+"_apphot_residuals.png")
        plt.close()

    return srclist