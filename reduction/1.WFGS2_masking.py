#==============================================
# BEFORE RUNNING
#==============================================
'''

This is the code for making the "Masking image" of FITS file for images taken by the wide field grism spectrograph 2 
(WFGS2; Uehara et al. 2004; Kawakami et al. 2022) on the 2.0-m Nayuta telescope at the Nishi–Harima Astronomical Observatory.
The "Masking image" masks the 1) nearby stars and 2) Cosmic ray.


1. 
 - Input file:  
   '*.fits'         Preprocessed FITS file with the WCS implemented
   '*.mag.1'        IRAF Phot file containing target's center info.
                    See below (i.e.,2. What you need to run this code)
 
 - Output file:
   'mask_*.fits'    Masking image in FITS format
   
   

2. What you need to run this code. The following packages must be installed.
  - astropy (https://www.astropy.org/)
  - Astro-SCRAPPY (https://github.com/astropy/astroscrappy)
  - "*.mag.1" file from IRAF's Phot package that contains the center of target's o-ray compnent.

3. 
In this code, the center of the target is found by using the phot of IRAF. So, we need the ".mag" file to bring the coordinate of target's ceter.
There is no problem if you find the target's center by other methods. 
All you need to do is modifying the part that brings the central coordinate of target.

  
4. Directory should contain the complete sets consist of 4 images (taken at HWP=0+90*n, 22.5+90*n, 45+90*n, 67.5+90n deg where n=0,1,2,3).
If the number of images in the directory is not a multiple of 4, an error occurs.
'''







#==============================================
# INPUT VALUE
#==============================================

Observatory = {'lon': 134.3356,
               'lat': 35.0253,
               'elevation': 0.449} #NHAO
Target_name = 3200
Obsdate_list  = 'The directory path where fits & mag.1 files are saved.'



#==============================================
# IMPORT PACKAGES AND DEFINE THE FUNCTION
#==============================================

import os
import glob 
import astropy
import astroquery
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import ascii
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter

import astroscrappy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from astroquery.jplhorizons import Horizons
from astropy.wcs import WCS
from astropy import units
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import warnings
from astropy.utils.exceptions import AstropyWarning
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
from tqdm import tqdm


def pill_masking(image,x1,x2,y1,y2,g_star,
                x_tar_init,y_tar_init,target_radi):
    
    x_star_str = x1
    x_star_end = x2
    y_star_str = y1
    y_star_end = y2
    
    Masking_image = np.zeros(np.shape(image))
    
    Y1 = max(0,int(min(y1)-150))
    Y2 = min(int(max(y2)+150),len(image))
    X1 = max(0,int(min(x1)-100))
    X2 = min(int(max(x2)+100),len(image[0]))
    
    
    for yi in np.arange(Y1,Y2,1):
        for xi in np.arange(X1,X2,1):
            for star in range(len(x_star_end)):
                if g_star[star] < 13:
                    height =25
                elif  13 <= g_star[star] <= 15:
                    height=19
                elif  15 < g_star[star] < 18:
                    height=16
                elif  18 <= g_star[star]:
                    height=15  
                slope = (y_star_end[star] - y_star_str[star])/(x_star_end[star]-x_star_str[star])
                theta = np.rad2deg(np.arctan(slope))
                modi_height = height / np.sin(np.deg2rad(90-theta))
                y_up = slope *xi + y_star_str[star] + modi_height - slope *x_star_str[star]
                y_low = slope *xi + y_star_str[star] - modi_height - slope *x_star_str[star]
                x_str = min(x_star_str[star],x_star_end[star])
                x_end = max(x_star_str[star],x_star_end[star])
                y_lower_cor = min(y_star_end[star],y_star_str[star])
                y_upper_cor = max(y_star_end[star],y_star_str[star])
                max_y = min(y_up,y_upper_cor+1)
                min_y = max(y_low,y_lower_cor)
                
                if (xi - x_star_str[star])**2 + (yi-y_star_str[star])**2 < (height)**2:
                    Masking_image[yi,xi] = 1
                if (xi - x_star_end[star])**2 + (yi-y_star_end[star])**2 < (height)**2:
                    Masking_image[yi,xi] = 1    
                if yi >= min_y and  yi <= max_y  and xi > x_str-height and xi < x_end+height: 
                    Masking_image[yi,xi] = 1    
                if  xi > x_str and xi < x_end and  yi < y_lower_cor+height and yi > y_lower_cor-height:
                    Masking_image[yi,xi] = 1    
                if (xi - x_tar_init)**2 + (yi-y_tar_init)**2 < (target_radi)**2:
                    Masking_image[yi,xi] = 0     
                    
                    
    return Masking_image

mpl.rc('figure', max_open_warning = 0)
np.set_printoptions(threshold=1000)
pd.set_option('display.max_rows', None)
warnings.simplefilter('ignore', category=AstropyWarning)

#==============================================
# BRING THE TARGET IMAGE TO BE MASKED
#==============================================

file = glob.glob(os.path.join(Obsdate_list,'w*Ph*.o.*.fits'))
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



#==============================================
# MAKE AND SAVE THE MASKING IMAGE
#==============================================


for n in range(len(file)):
    fi = file[n]
    if os.path.exists(Obsdate_list+'/mask_'+fi.split('/')[-1]) ==True:
        print(Obsdate_list+'/mask_'+fi.split('/')[-1],' already exists.')
        continue
    hdul = fits.open(fi)[0]
    header = hdul.header
    OBJECT = header['OBJECT']
    if OBJECT != 'Phaethon':
        print('No phaethon',fi)
        continue
    data_o = hdul.data
    gain = header['GAIN']
    RN = header['RDNOISE']
    JD = header['MJD-CEN'] + 2400000.5#Central MJD
    
    fi_ = fi.replace('.o.','.e.')
    hdul_e = fits.open(fi_)[0]
    data_e = hdul_e.data
    
    
    obj = Horizons(id=Target_name,location=Observatory,epochs=JD)
    eph = obj.ephemerides()
    ra_mid,dec_mid = eph['RA'][0], eph['DEC'][0] 
    
    ##
    Center = ascii.read(fi+'.mag.1')
    header['CRPIX1'], header['CRPIX2'] = Center['XCENTER'][0], Center['YCENTER'][0]
    header['CRVAL1'], header['CRVAL2'] = ra_mid,dec_mid
    
    
    w =WCS(header)
    x_tar_init,y_tar_init = w.wcs_world2pix(ra_mid,dec_mid,0)
        

    #Find the center of target
    index = 30
    y_1, y_2 = int(y_tar_init-index), int(y_tar_init+index)
    x_1, x_2 = int(x_tar_init-index), int(x_tar_init+index)
    
    if x_2 > len(data_o[0]):
        x_2 = len(data_o[0])
    
    crop_o_ = data_o[y_1:y_2,x_1:x_2]
    crop_o = crop_o_ - np.median(crop_o_[:5])
    y, x = np.mgrid[:len(crop_o), :len(crop_o[0])]
    g_init = Gaussian2D(x_mean = index,y_mean=index,
                        theta=0,
                        amplitude=crop_o[index,index],
                        bounds={'x_mean':(index-30,index+30),
                                'y_mean':(index-30,index+30)})

    fitter = LevMarLSQFitter()
    fitted = fitter(g_init, x,y, crop_o)
    center_x = fitted.x_mean.value
    center_y = fitted.y_mean.value
    fwhm = max(fitted.x_fwhm,fitted.y_fwhm)
    x_tar,y_tar = center_x+x_1 , center_y + y_1
    
    
    
    
    
    ##############################################
    #           MAKE THE MASKED IMAGE
    ##############################################
#         Find the background stars's RA,DEC from Gaia
    coord = SkyCoord(ra=ra_mid, dec=dec_mid, unit=(units.degree, units.degree), frame='icrs')
    width = units.Quantity(0.02, units.deg)
    height = units.Quantity(0.02, units.deg)
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    RA_star = []
    DEC_star = []
    g_star = []
    for i in range(len(r)):
        if r[i]['phot_g_mean_mag']<21:
            RA_star.append(r[i]['ra'])
            DEC_star.append(r[i]['dec'])
            g_star.append(r[i]['phot_g_mean_mag'])

    #==================================#        
    # Convert ra,dec to x,y of Star    #
    #==================================#
    #target ra, dec when exp start
    JD_str = header['MJD-STR'] + 2400000.5
    obj = Horizons(id=Target_name,location=Observatory,epochs=JD_str)
    eph = obj.ephemerides()
    ra_str,dec_str = eph['RA'][0], eph['DEC'][0]
    header_str = header
    header_str['CRVAL1'], header_str['CRVAL2'] = ra_str,dec_str
    header_str['CRPIX1'], header_str['CRPIX2'] = float(x_tar_init), float(y_tar_init)


    X_str = []
    Y_str = [] 
    w_str =WCS(header_str)
    for i in range(len(RA_star)):
        x_str_star,y_str_star = w_str.wcs_world2pix(RA_star[i],DEC_star[i],0)
        X_str.append(float(x_str_star))
        Y_str.append(float(y_str_star))



    #target ra, dec when exp end
    JD_end = header['MJD-END'] + 2400000.5
    obj = Horizons(id=Target_name,location=Observatory,epochs=JD_end)
    eph = obj.ephemerides()
    ra_end,dec_end = eph['RA'][0], eph['DEC'][0]   
    header_end = header
    header_end['CRVAL1'], header_end['CRVAL2'] = ra_end,dec_end 
    header_end['CRPIX1'], header_end['CRPIX2'] = float(x_tar_init), float(y_tar_init)

    X_end = []
    Y_end = []    

    w_end =WCS(header_end)
    for i in range(len(RA_star)):
        x_end_star,y_end_star = w_end.wcs_world2pix(RA_star[i],DEC_star[i],0)
        X_end.append(float(x_end_star))
        Y_end.append(float(y_end_star))

    Masking_image_str = pill_masking(data_o,X_str,X_end,Y_str,Y_end,g_star,
                                    x_tar_init,y_tar_init,8)


    MASK_o = Masking_image_str  
    MASK_e = Masking_image_str.copy()

    #Bad pixel remove
    m_LA_o,cor_image_o = astroscrappy.detect_cosmics(data_o,
                                                 gain = gain,
                                                 readnoise = RN,
                                                 sigclip=5)
    tmLA_o = m_LA_o.astype(int)
    MASK_o[tmLA_o == 1 ] = 2


    m_LA_e,cor_image_e = astroscrappy.detect_cosmics(data_e,
                                                     gain = gain,
                                                     readnoise = RN,
                                                     sigclip=5)
    tmLA_e = m_LA_e.astype(int)
    MASK_e[tmLA_e == 1 ] = 2


    filename_o = Obsdate_list+'/mask_'+fi.split('/')[-1]
    fits.writeto(filename_o,data=MASK_o,header=header,overwrite=True)
    print(filename_o +' is created.')


    filename_e = filename_o.replace('.o.','.e.')
    fits.writeto(filename_e,data=MASK_e,header=header,overwrite=True)
    print(filename_e +' is created.')



    figsize=60
    fig,ax = plt.subplots(1,2,figsize=(15,6))
    masked_image = np.ma.masked_array(data_o-np.median(crop_o_[:5]),MASK_o)
    im = ax[0].imshow(masked_image,vmin=-100,vmax=100,cmap='seismic')
    ax[0].set_xlim(x_tar-figsize,x_tar+figsize)
    ax[0].set_ylim(y_tar-figsize,y_tar+figsize)
    ax[0].set_title(fi.split('/')[-1],fontsize=18)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)

    crop_e_ = data_e[y_1:y_2,x_1:x_2]
    masked_image_e = np.ma.masked_array(data_e-np.median(crop_e_[:5]),MASK_e)
    im2 = ax[1].imshow(masked_image_e,vmin=-100,vmax=100,cmap='seismic')
    ax[1].set_xlim(x_tar-figsize,x_tar+figsize)
    ax[1].set_ylim(y_tar-figsize,y_tar+figsize)
    ax[1].set_title(fi_.split('/')[-1],fontsize=18)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2,cax=cax)
    plt.show()

