
#==============================================
# BEFORE RUNNING
#==============================================

'''
This is the code for making the aperture photometry of the asteroids taken by PICO (Ikeda et al. 2004).


1. 
 - Input file:  
   '*.fits'                     Preprocessed FITS file with the WCS implemented / each FITS file contain only one component (o-ray or e-ray)
   '*.mag.1'                    IRAF Phot file containing target's center info.
                                See below (i.e.,2. What you need to run this code)
   'mask_*.fits'                Masking image produced by '1.HONIR_masking.py'.  
 
 
 - Outout file:
   Phot_{DATE}_{Object_name}.csv         Photometric result of each images 
             

2. What you need to run this code. The following packages must be installed.
  - astropy (https://www.astropy.org/)
  - "*.mag.1" file from IRAF's Phot package that contains the center of the target's o-ray (or e-ray) component.
  

3. 
In this code, the center of the target is found by using the phot of IRAF. 
So, we need the ".mag" file to bring the coordinate of the target's center.
There is no problem if you find the target's center by other methods. 
All you need to do is modify the part that brings the central coordinate of the target.

  
4. Directory should contain the complete sets consisting of 8 images 
(o-ray and e-ray in which each taken at HWP=0+90*n, 22.5+90*n, 45+90*n, 67.5+90n deg where n=0,1,2,3).
If the number of images in the directory is not a multiple of 8, an error occurs.
'''

#==============================================
# INPUT VALUE
#==============================================
Target_name = 3200
Observatory = {'lon': 139.56,
               'lat': 35.67,
               'elevation': 0.01} #NAOJ
RN = 15 #from https://www.osn.iaa.csic.es/sites/default/files/documents/stl11000m_catalog.pdf
subpath  = 'The directory path where fits & *.mag.1 & mask_*.fits files are saved.'


####################################
# Photometry Parameter
####################################
#Values below are examples. Different values were used for each data.
Aperture_scale = 1.8    # Aperture radius = Aperture_scale * FWHM 
ANN_scale = 4         # Annulus radius = ANN_scale * FWHM
Dan = 20          # [pix] #Dannulus size


fig_plot = 'yes' #Will you plot the image? or 'No'






#==============================================
# IMPORT PACKAGES AND DEFINE THE FUNCTION
#==============================================

import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import warnings

from astropy.io import fits, ascii
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import gaussian_fwhm_to_sigma

def skyvalue(data, y0, x0, r_in, r_out,masking=None):
    if masking is not None:
        masking = masking.astype(bool)
    else:
        masking = np.zeros(np.shape(data))

    # Determine sky and std
    y_in = int(y0-r_out)
    y_out = int(y0+r_out)
    x_in = int(x0-r_out)
    x_out = int(x0+r_out)
    if y_in < 0:
        y_in = 0
    if y_out > len(data) :
        y_out = len(data)
    if x_in < 0:
        x_in = 0
    if x_out > len(data[0]):
        x_out =  len(data[0])
        
    sky_deriving_area = data[y_in:y_out, x_in:x_out]
    masking = masking[y_in:y_out, x_in:x_out]
    
    new_mask = np.zeros(np.shape(sky_deriving_area))+1
    for yi in range(len(sky_deriving_area)):
        for xi in range(len(sky_deriving_area[0])):
            position = (xi - r_out)**2 + (yi-r_out)**2
            if position < (r_out)**2 and position > r_in**2:
                new_mask[yi, xi] = 0
    new_mask = new_mask.astype(bool)
    mask = new_mask + masking
    
    Sky_region = np.ma.masked_array(sky_deriving_area, mask)
    std = np.ma.std(Sky_region)
    sky = np.ma.median(Sky_region)
    npix = np.shape(sky_deriving_area)[0]*np.shape(sky_deriving_area)[1] - np.sum(mask)
    
    return(sky, std, npix)

def signal_to_noise(source_eps, sky_eps, rd, npix,
                            gain):
    signal = source_eps 
    noise = np.sqrt((source_eps  + npix *
                         (sky_eps * gain )) + npix * rd ** 2)
    return signal / noise   

def circle(x,y,r):
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = r*np.cos(theta)+y
    x2 = r*np.sin(theta)+x
    return(x2.tolist(),x1.tolist())


#==============================================
# BRING THE TARGET IMAGE
#==============================================

file = glob.glob(os.path.join(subpath,'*.fits'))
file_= []
for fi in file:
    if 'mask' not in fi:
        file_.append(fi)
file = sorted(file_)

log = pd.DataFrame({})
for fi in file:
    hdul = fits.open(fi)
    header = hdul[0].header
    data = hdul[0].data
    log = log.append({'FILENAME':os.path.split(fi)[-1],
                      'DATE':header['DATE-OBS'],
                      'EXPTIME':header['EXPTIME']},
                      ignore_index=True)
print(log)


#======================================#
#             Photometry               #
#======================================#
Photo_Log = pd.DataFrame({})
order = np.arange(0,len(file),8)
for z in order:
    SET = [file[z],file[z+1], file[z+2], file[z+3],file[z+4],file[z+5], file[z+6], file[z+7]]
    for i in range(0,8):
        RET = SET[i]  #Bring the fits file
        print(RET)
        hdul = fits.open(RET)
        header = hdul[0].header 
        data = hdul[0].data
        gain = header['EGAIN']
        JD = header['JD_CEN'] #Central JD
        DATE = header['DATE-OBS'].split('T')[0]
        UT = header['DATE-OBS'].split('T')[1]
        
        
        #===============================================
        # Bring the mask image
        #===============================================
        MASK_hdul = fits.open(os.path.join(subpath,'mask_'+RET.split('/')[-1]))[0]
        masking = MASK_hdul.data
        masked_data = np.ma.masked_array(data,masking)
        
        
        #===============================================
        # Bring ephemeride info.
        #===============================================
        obj = Horizons(id=Target_name,location=Observatory,epochs=JD)
        eph = obj.ephemerides()
        ra_mid,dec_mid = eph['RA'][0], eph['DEC'][0]   
        psANG_i = eph['sunTargetPA'][0] #[deg]
        pA_i = eph['alpha'][0] #[deg]    
        delta_i = eph['delta'][0]
        r_i = eph['r'][0]
        airmass_i = eph['airmass'][0]
        EL_i = eph['EL'][0]
        AZ_i = eph['AZ'][0]

        #===============================================
        # Bring the center of target
        #===============================================
        magfile = ascii.read(RET+'.mag.1')
        x_tar_init,y_tar_init = magfile['XCENTER'][0],magfile['YCENTER'][0]
        
        
        #===============================================
        # Determined the FWHM & Updated the center
        #===============================================
        index = 50

        #Ordinary
        lim = 20
        y_1, y_2 = int(y_tar_init-index), int(y_tar_init+index)
        x_1, x_2 = int(x_tar_init-index), int(x_tar_init+index)

        crop = data[y_1:y_2,x_1:x_2]
        crop_mask = masking[y_1:y_2,x_1:x_2]
        crop_sub = crop - skyvalue(data, y_tar_init, x_tar_init, 15, 20,masking=masking)[0]
        y, x = np.mgrid[:len(crop), :len(crop[0])]
        g_init = Gaussian2D(x_mean = index,y_mean=index,
                            theta=0,
                            amplitude=crop_sub[index,index],
                            bounds={'x_mean':(index-lim,index+lim),
                                    'y_mean':(index-lim,index+lim)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x,y, crop_sub)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        fwhm = max(fitted.x_fwhm,fitted.y_fwhm)
        x_tar, y_tar = center_x+x_1, center_y+y_1
        
        
        
        #===============================================
        # APERTURE PHOTOMETRY
        #===============================================
        #### Set aperture size
        Aperture_radius = Aperture_scale*fwhm/2
        Ann = ANN_scale*fwhm/2
        Ann_out = Ann+Dan
        
        
        ##Determine sky value by aperture   
        Aper = CircularAperture([x_tar,y_tar],Aperture_radius) #Set aperture
        sky,std,area = skyvalue(data,y_tar,x_tar,Ann,Ann_out,masking) # Set area determinung Sk

        Flux = aperture_photometry(data - sky,Aper,masking)['aperture_sum'][0]*gain
        ERR = np.sqrt(Flux + 3.14*Aperture_radius**2*(sky*gain + (std*gain)**2 +(RN*gain)**2))
        Snr = signal_to_noise(Flux,sky,RN,Aperture_radius**2*3.14,gain)
        
        if '.o.' in RET:
            ray = 'o'
        elif '.e.' in RET:
            ray = 'e'
        
        
        Photo_Log = Photo_Log.append({'filename':RET.split('/')[-1],
                                      'DATE':DATE,
                                      'UT':UT,
                                      'Filter':'Rc',
                                      'Object':Target_name,
                                      'JD':JD,
                                      'ray':ray,
                                      'alpha [deg]':pA_i,
                                      'PsANG [deg]':psANG_i,
                                      'FWHM [pix]': fwhm,
                                      'Aper_radius [pix]':Aperture_radius,
                                     'Flux [e]':Flux,
                                     'eFlux [e]':ERR,
                                     'Sky [e]':sky,
                                     'eSky [e]': std,
                                      'SNR':Snr,
                                      'delta':delta_i,
                                      'r':r_i,
                                      'Airmass':airmass_i
                                     },
                                    ignore_index=True)
        
        
        
        
        
        
        if fig_plot == 'yes':
            lim = 200
            fig,ax = plt.subplots(1,1,figsize=(8,8))
            plot_data = np.ma.masked_array(data,masking)
            figsize=100
            im = ax.imshow(plot_data - sky,vmin=-lim,vmax=lim,cmap='seismic')
            xi,yi = circle(x_tar,y_tar,Aperture_radius)
            ax.plot(xi,yi,color='y',lw=4)
            xi,yi = circle(x_tar,y_tar,Ann)
            ax.plot(xi,yi ,color='c',lw=4)
            xi,yi = circle(x_tar,y_tar,Ann+Dan)
            ax.plot(xi,yi ,color='c',lw=4)
            ax.plot(x_tar,y_tar,marker='+',ls='',color='b')
            ax.set_xlim(x_tar-figsize,x_tar+figsize)
            ax.set_ylim(y_tar-figsize,y_tar+figsize)
            ax.set_title(RET.split('/')[-1],fontsize=18)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im,cax=cax) 
            plt.show()
            
            
new_index = ['filename','DATE','UT','Filter','Object','JD','ray',
             'alpha [deg]','Flux [e]','eFlux [e]',
             'Sky [e]','eSky [e]','SNR','delta','r',
             'PsANG [deg]','FWHM [pix]',
             'Aper_radius [pix]','Airmass']           
Photo_Log = Photo_Log.reindex(columns = new_index)  
Photo_Log = Photo_Log.round({'alpha [deg]':2,'PsANG [deg]':2,'FWHM [pix]':1,
                             'Aper_radius [pix]':1,'Flux [e]':2,'eFlux [e]':2,
                             'Sky [e]':2,'eSky [e]':2,'SNR':1,'delta':1,'r':1,'Airmass':1})
DATE = DATE.replace('-','_')
FILENAME = os.path.join(subpath,'Phot_{0}_{1}.csv'.format(DATE,Target_name))
Photo_Log.to_csv(FILENAME) 


