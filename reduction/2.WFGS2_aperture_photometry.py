


#==============================================
# BEFORE RUNNING
#==============================================

'''
This is the code for the aperture photometry of the asteroid target taken by the wide field grism spectrograph 2 
(WFGS2; Uehara et al. 2004; Kawakami et al. 2022) on the 2.0-m Nayuta telescope at the Nishi–Harima Astronomical Observatory.


1. 
 - Input file:  
   '*.fits'         Preprocessed FITS file
   '*.mag.1'        IRAF Phot file containing target's center info.
                    See below (i.e.,2. What you need to run this code)
   'mask_*.fits'    Masking image produced by '1.WFGS2_maksing.py'.   
 
 - Outout file:
   Phot_{DATE}_{Object_name}.csv         Photometric result of each images 
             

2. What you need to run this code. The following packages must be installed.
  - astropy (https://www.astropy.org/)
  - "*.mag.1" file from IRAF's Phot package that contains the center of target's o-ray compnent.


3. 
In this code, the center of the target is found by using the phot of IRAF. 
So, we need the ".mag" file to bring the coordinate of target's ceter.
There is no problem if you find the target's center by other methods.
All you need to do is modifying the part that brings the central coordinate of target.

4. Directory should contain the complete sets consist of 4 images (taken at HWP=0+90*n, 22.5+90*n, 45+90*n, 67.5+90n deg where n=0,1,2,3).
If the number of images in the directory is not a multiple of 4, an error occurs.
'''





#==============================================
# INPUT VALUE FOR THE APERTURE PHOTOMETRY
#==============================================

Obsdate_list  = 'Directory path where fits & *.mag.1 & mask_*.fits files are saved.' 
Observatory = {'lon': 134.3356,
               'lat': 35.0253,
               'elevation': 0.449} #NHAO
Target_name = 3200

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

import glob 
import os
import astropy
import photutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.io import ascii
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from astroquery.jplhorizons import Horizons
from astropy.wcs import WCS
from astropy import units
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

from photutils import CircularAperture,CircularAnnulus,aperture_photometry
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sep

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', UserWarning)



    
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

    
def circle(x,y,r):
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = r*np.cos(theta)+y
    x2 = r*np.sin(theta)+x
    return(x2.tolist(),x1.tolist())   
    
def signal_to_noise(source_eps, sky_eps, rd, npix,
                            gain):
    signal = source_eps * gain
    noise = np.sqrt( (source_eps * gain + npix *
                         (sky_eps * gain )) + npix * rd ** 2)
    return signal / noise      



#==============================================
# BRING THE TARGET IMAGE
#==============================================

file = glob.glob(os.path.join(Obsdate_list,'w*Ph*.*.fits'))
file = sorted(file)

log = pd.DataFrame({})
for fi in file:
    hdul = fits.open(fi)
    header = hdul[0].header
    data = hdul[0].data
    log = log.append({'FILENAME':os.path.split(fi)[-1],
                      'OBJECT':header['OBJECT'],
                      'DATE':header['DATE-OBS'],
                      'HWPANG':header['HWP-AGL'],
                      'EXPTIME':header['EXPTIME']},
                      ignore_index=True)


#======================================#
#             Photometry               #
#======================================#

Photo_Log = pd.DataFrame({})

for i in range(len(file)//2):
    fi_e = file[2*i]
    fi_o = file[2*i+1]
    
    hdul_o = fits.open(fi_o)[0]
    header_o = hdul_o.header
    data_o = hdul_o.data
    
    hdul_e = fits.open(fi_e)[0]
    header_e = hdul_e.header
    data_e = hdul_e.data

    OBJECT = header_o['OBJECT']
    gain = 2.28#header['GAIN']
    RN = 13.8#header['RDNOISE']
    JD = header_o['MJD-CEN'] + 2400000.5#Central MJD

    #Bring ephemeride info.
    obj = Horizons(id=Target_name,location=Observatory,epochs=JD)
    eph = obj.ephemerides()
    ra_mid,dec_mid = eph['RA'][0], eph['DEC'][0]   
    psANG = eph['sunTargetPA'][0] 
    pA = eph['alpha'][0] 
    
    #============================================
    # ORDINARY
    #===========================================

    w =WCS(header_o)  
    x_tar_init,y_tar_init = w.wcs_world2pix(ra_mid,dec_mid,0)
        
    Cen = ascii.read(fi_o+'.mag.1')    
    x_tar_init, y_tar_init = Cen['XCENTER'][0], Cen['YCENTER'][0]


    #Bring masking image 
    MASK_hdul_o = fits.open(Obsdate_list+'/mask_'+fi_o.split('/')[-1])[0]
    masking_o = MASK_hdul_o.data
    maksing_o = (masking_o).astype(bool)        
    
    #Find the center of target
    masked_data_o = np.ma.masked_array(data_o,maksing_o)

    #===============================================
    # Determined the FWHM
    #===============================================
    index = 70
    y_1, y_2 = int(y_tar_init-index), int(y_tar_init+index)
    x_1, x_2 = int(x_tar_init-index), int(x_tar_init+index)

    crop_o = masked_data_o[y_1:y_2,x_1:x_2]
    sky_,std__,area__ = skyvalue(data_o,y_tar_init,x_tar_init,30,40,maksing_o)
    
    crop_o_ = crop_o - sky_
    y, x = np.mgrid[:len(crop_o), :len(crop_o[0])]
    g_init = Gaussian2D(x_mean = index,
                        y_mean = index,
                        theta = 0,
                        amplitude = crop_o_[index,index],
                        bounds = {'x_mean':(index-15,index+15),
                                  'y_mean':(index-15,index+15)})
    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x,y, crop_o_)
    center_x = fitted.x_mean.value
    center_y = fitted.y_mean.value
    g_init = Gaussian2D(x_mean = center_x,
                        y_mean = center_y,
                        theta = 0,
                        amplitude = crop_o_[index,index],
                        bounds = {'x_mean':(index-15,index+15),
                                  'y_mean':(index-15,index+15)})
    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x, y, crop_o_)
    center_x = fitted.x_mean.value
    center_y = fitted.y_mean.value
    fwhm_o = max(fitted.x_fwhm,fitted.y_fwhm)
    x_tar_o, y_tar_o =  center_x, center_y
        
        
        
    #============================================
    # EXTRA-ORDINARY
    #===========================================
    #Bring masking image 
    MASK_hdul_e = fits.open(Obsdate_list+'/mask_'+fi_e.split('/')[-1])[0]
    masking_e = MASK_hdul_e.data
    maksing_e = (masking_e).astype(bool)        
    
    #Find the center of target
    masked_data_e = np.ma.masked_array(data_e,maksing_e)

    #===============================================
    # Determined the FWHM
    #===============================================

    index = 70
    y_1, y_2 = int(y_tar_init-index), int(y_tar_init+index)
    x_1, x_2 = int(x_tar_init-index), int(x_tar_init+index)

    crop_e = masked_data_e[y_1:y_2,x_1:x_2]
    sky_,std__,area__ = skyvalue(data_e,y_tar_init,x_tar_init,30,40,maksing_e)
    
    crop_e_ = crop_e - sky_
    y, x = np.mgrid[:len(crop_e), :len(crop_e[0])]
    g_init = Gaussian2D(x_mean = index,y_mean=index,
                        theta=0,
                        amplitude=crop_e_[index,index],
                        bounds={'x_mean':(index-15,index+15),
                                'y_mean':(index-15,index+15)})
    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x,y, crop_e_)
    center_x = fitted.x_mean.value
    center_y = fitted.y_mean.value
    g_init = Gaussian2D(x_mean = center_x,y_mean=center_y,
                        theta=0,
                        amplitude=crop_e_[index,index],
                        bounds={'x_mean':(index-15,index+15),
                                'y_mean':(index-15,index+15)})
    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x, y, crop_e_)
    center_x = fitted.x_mean.value
    center_y = fitted.y_mean.value
    fwhm_e = max(fitted.x_fwhm,fitted.y_fwhm)
    x_tar_e, y_tar_e =  center_x, center_y
  
    fwhm = max(fwhm_o, fwhm_e)
    
    
    #===============================================
    # Set Aperture size & Ann
    #===============================================
    #Aperture Photometry
    Aperture_radius = fwhm*Aperture_scale/2
    Ann =  fwhm*ANN_scale/2
    Ann_out = Ann+ Dan
    
    #===============================================
    # Replace the bad pixel by interpolating (only in the aperture)
    #===============================================
    mask_o = (crop_o.mask).astype(bool)
    mask_temp_o = np.copy(mask_o)
    mask_temp_o = (mask_temp_o).astype(bool)
    for yi in range(len(mask_temp_o)):
        for xi in range(len(mask_temp_o[0])):
            if (xi - x_tar_o)**2 + (yi-y_tar_o)**2 > (Aperture_radius)**2:    
                mask_temp_o[yi,xi] = 0   
    new_o = data_o[y_1:y_2,x_1:x_2]
    
    new_o = new_o.byteswap().newbyteorder()
    bkg_o = sep.Background(new_o, mask=mask_o, bw=2, bh=2, fw=1, fh=1)
    bkg_image_o = bkg_o.back()      
    Target_masking_o= (mask_temp_o).astype(bool)
    new_o[Target_masking_o] = bkg_image_o[Target_masking_o]
    

    mask_e = (crop_e.mask).astype(bool)
    mask_temp_e = np.copy(mask_e)
    mask_temp_e = (mask_temp_e).astype(bool)
    for yi in range(len(mask_temp_e)):
        for xi in range(len(mask_temp_e[0])):
            if (xi - x_tar_e)**2 + (yi-y_tar_e)**2 > (Aperture_radius)**2:    
                mask_temp_e[yi,xi] = 0   
    new_e = data_e[y_1:y_2,x_1:x_2]    
    
    new_e = new_e.byteswap().newbyteorder()
    bkg_e = sep.Background(new_e, mask=mask_e, bw=2, bh=2, fw=1, fh=1)
    bkg_image_e = bkg_e.back()      
    Target_masking_e= (mask_temp_e).astype(bool)
    new_e[Target_masking_e] = bkg_image_e[Target_masking_e]    

        
    
    #===============================================
    # Modifying the masking image
    #===============================================
    modi_mask_o = np.copy(mask_o)
    for yi in range(len(mask_o)):
        for xi in range(len(mask_o[0])):
            if (xi - x_tar_o)**2 + (yi-y_tar_o)**2 < (Aperture_radius)**2:    
                modi_mask_o[yi,xi] = 0   
                
    modi_mask_e = np.copy(mask_e)
    for yi in range(len(mask_e)):
        for xi in range(len(mask_e[0])):
            if (xi - x_tar_e)**2 + (yi-y_tar_e)**2 < (Aperture_radius)**2:    
                modi_mask_e[yi,xi] = 0    
    
    #===============================================
    # Aperture Photometry
    #===============================================
    Aper_o = CircularAperture([x_tar_o,y_tar_o],Aperture_radius) #Set aperture
    sky_o,std_o,area_o = skyvalue(new_o,y_tar_o,x_tar_o,Ann,Ann_out,modi_mask_o) # Set area determinung Sksky,std,area = skyvalue(data,y_tar,x_tar,Ann,Ann_out,masking) # Set area determinung Sk
       
    Flux_o = aperture_photometry(new_o - sky_o,Aper_o,modi_mask_o)['aperture_sum'][0]*gain
    ERR_o = np.sqrt(Flux_o + 3.14*Aperture_radius**2*(sky_o*gain + std_o**2 +(RN*gain)**2))
    Snr_o = signal_to_noise(Flux_o,sky_o,RN,Aperture_radius**2*3.14,gain)        
    
    Aper_e = CircularAperture([x_tar_e,y_tar_e],Aperture_radius) #Set aperture
    sky_e,std_e,area_e = skyvalue(new_e,y_tar_e,x_tar_e,Ann,Ann_out,modi_mask_e) # Set area determinung Sksky,std,area = skyvalue(data,y_tar,x_tar,Ann,Ann_out,masking) # Set area determinung Sk
       
    Flux_e = aperture_photometry(new_e - sky_e,Aper_e,modi_mask_e)['aperture_sum'][0]*gain
    ERR_e = np.sqrt(Flux_e + 3.14*Aperture_radius**2*(sky_e*gain + std_e**2 +(RN*gain)**2))
    Snr_e = signal_to_noise(Flux_e,sky_e,RN,Aperture_radius**2*3.14,gain)        


        
    maksing_o = (masking_o[y_1:y_2,x_1:x_2]).astype(bool)
    maksing_e = (masking_e[y_1:y_2,x_1:x_2]).astype(bool)
    danger_score_o = aperture_photometry(maksing_o,Aper_o)['aperture_sum'][0]
    danger_score_e = aperture_photometry(maksing_e,Aper_e)['aperture_sum'][0]
    danger_score = danger_score_o+danger_score_e
    
    if danger_score == 0:
        danger_ref = 1     #Reliable result
    elif 10 > danger_score > 0:
        danger_ref = 2    #Pixel values obtained by interpolation are used in the results.  
    elif danger_score > 10:
        danger_ref = 3    #Be careful: More than 10 pixel values obtained by interpolation are used in the result
        
        
    if fig_plot == 'yes':
        figsize=70
        masked_o = np.ma.masked_array(new_o,modi_mask_o)
        fig,ax = plt.subplots(1,3,figsize=(18,6))
        im = ax[0].imshow(masked_o - sky_o,vmin=-100,vmax=100,cmap='seismic')
        xi,yi = circle(x_tar_o,y_tar_o,Aperture_radius)
        ax[0].plot(xi,yi,color='k',lw=2)
        xi,yi = circle(x_tar_o,y_tar_o,Ann)
        ax[0].plot(xi,yi ,color='k',lw=2,ls='--')
        xi,yi = circle(x_tar_o,y_tar_o,Ann_out)
        ax[0].plot(xi,yi ,color='k',lw=2,ls='--')
        ax[0].plot(x_tar_o,y_tar_o,marker='+',ls='',color='b')

        ax[0].set_xlim(x_tar_o-figsize,x_tar_o+figsize)
        ax[0].set_ylim(y_tar_o-figsize,y_tar_o+figsize)
        ax[0].set_title(fi_o.split('/')[-1],fontsize=18)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im,cax=cax)

        im = ax[1].imshow(new_o, vmin = np.median(crop_o)*0.8, vmax = np.median(crop_o)*1.2)
        xi,yi = circle(x_tar_o,y_tar_o,Aperture_radius)
        ax[1].plot(xi,yi,color='w',lw=2)
        xi,yi = circle(x_tar_o,y_tar_o,Ann)
        ax[1].plot(xi,yi ,color='w',lw=2,ls='--')
        xi,yi = circle(x_tar_o,y_tar_o,Ann_out)
        ax[1].plot(xi,yi ,color='w',lw=2,ls='--')
        ax[1].plot(x_tar_o,y_tar_o,marker='+',ls='',color='w')
        ax[1].set_xlim(x_tar_o-figsize,y_tar_o+figsize)
        ax[1].set_ylim(x_tar_o-figsize,y_tar_o+figsize)
        ax[1].set_title('Image with interpolated pixel values',fontsize=10)

        im = ax[2].imshow(maksing_o)
        xi,yi = circle(x_tar_o,y_tar_o,Aperture_radius)
        ax[2].plot(xi,yi,color='w',lw=2)
        ax[2].plot(x_tar_o,y_tar_o,marker='+',ls='',color='w')
        ax[2].set_xlim(x_tar_o-figsize,y_tar_o+figsize)
        ax[2].set_ylim(x_tar_o-figsize,y_tar_o+figsize)
        ax[2].set_title('Masking image, Flag:{0}'.format(danger_score),fontsize=13)    


        figsize=70
        masked_e = np.ma.masked_array(new_e,modi_mask_e)
        fig,ax = plt.subplots(1,3,figsize=(18,6))
        im = ax[0].imshow(masked_e - sky_e,vmin=-100,vmax=100,cmap='seismic')
        xi,yi = circle(x_tar_e,y_tar_e,Aperture_radius)
        ax[0].plot(xi,yi,color='k',lw=2)
        xi,yi = circle(x_tar_e,y_tar_e,Ann)
        ax[0].plot(xi,yi ,color='k',lw=2,ls='--')
        xi,yi = circle(x_tar_e,y_tar_e,Ann_out)
        ax[0].plot(xi,yi ,color='k',lw=2,ls='--')
        ax[0].plot(x_tar_e,y_tar_e,marker='+',ls='',color='b')

        ax[0].set_xlim(x_tar_e-figsize,x_tar_e+figsize)
        ax[0].set_ylim(y_tar_e-figsize,y_tar_e+figsize)
        ax[0].set_title(fi_e.split('/')[-1],fontsize=18)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im,cax=cax)

        im = ax[1].imshow(new_e, vmin = np.median(crop_e)*0.8, vmax = np.median(crop_e)*1.2)
        xi,yi = circle(x_tar_e,y_tar_e,Aperture_radius)
        ax[1].plot(xi,yi,color='w',lw=2)
        xi,yi = circle(x_tar_e,y_tar_e,Ann)
        ax[1].plot(xi,yi ,color='w',lw=2,ls='--')
        xi,yi = circle(x_tar_e,y_tar_e,Ann_out)
        ax[1].plot(xi,yi ,color='w',lw=2,ls='--')
        ax[1].plot(x_tar_e,y_tar_e,marker='+',ls='',color='w')
        ax[1].set_xlim(x_tar_e-figsize,y_tar_e+figsize)
        ax[1].set_ylim(x_tar_e-figsize,y_tar_e+figsize)
        ax[1].set_title('Image with interpolated pixel values',fontsize=10)

        im = ax[2].imshow(maksing_e)
        xi,yi = circle(x_tar_e,y_tar_e,Aperture_radius)
        ax[2].plot(xi,yi,color='w',lw=2)
        ax[2].plot(x_tar_e,y_tar_e,marker='+',ls='',color='w')
        ax[2].set_xlim(x_tar_e-figsize,y_tar_e+figsize)
        ax[2].set_ylim(x_tar_e-figsize,y_tar_e+figsize)
        ax[2].set_title('Masking image, Flag:{0}'.format(danger_score),fontsize=13)    
    

    Photo_Log = Photo_Log.append({'filename':fi_e.split('/')[-1],
                                  'Object':header_e['OBJECT'],
                                  'DATE':header_e['DATE-OBS'],
                                  'JD':JD,
                                  'HWPANG':header_e['HWP-AGL'],
                                  'ray':'e',
                                  'Filter':header_e['FILTER02'],
                                  'Flux':Flux_e,
                                  'eFlux':ERR_e,
                                  'PA':pA,
                                  'PsANG':psANG,
                                  'SNR':Snr_e,
                                  'Aper_pix':Aperture_radius,
                                  'sky':sky_e,
                                  'esky':std_e,
                                  'level':danger_ref,
                                  'EXPTIME':header_e['EXPTIME'],
                                  'INST-PA':header_e['INST-PA'],
                                  'INSROT':header_e['INSROT']},
                                 ignore_index=True)

    Photo_Log = Photo_Log.append({'filename':fi_o.split('/')[-1],
                                  'Object':header_o['OBJECT'],
                                  'DATE':header_o['DATE-OBS'],
                                  'JD':JD,
                                  'HWPANG':header_o['HWP-AGL'],
                                  'ray':'o',
                                  'Filter':header_o['FILTER02'],
                                  'Flux':Flux_o,
                                  'eFlux':ERR_o,
                                  'PA':pA,
                                  'PsANG':psANG,
                                  'SNR':Snr_o,
                                  'Aper_pix':Aperture_radius,
                                  'sky':sky_o,
                                  'esky':std_o,
                                  'level':danger_ref,
                                  'EXPTIME':header_o['EXPTIME'],
                                  'INST-PA':header_o['INST-PA'],
                                  'INSROT':header_o['INSROT']},
                                 ignore_index=True)
   


new_index = ['filename', 'Object','DATE','JD','HWPANG','EXPTIME',
             'ray','Filter','Flux','eFlux','sky','esky',
             'PA','PsANG','INST-PA','INSROT','SNR','Aper_pix','level']
Photo_Log = Photo_Log.reindex(columns = new_index)      
Photo_Log = Photo_Log.round({'Flux': 1, 'eFlux': 1,'SNR': 1,'sky': 1,
                            'esky': 0 ,'JD':10})

filename = os.path.join(Obsdate_list,'Phot_{0}_{1}.csv'.format(header_o['DATE-OBS'].replace('-','_'),header_o['OBJECT']))
Photo_Log.to_csv(filename)    
print(filename + ' is saved.')

