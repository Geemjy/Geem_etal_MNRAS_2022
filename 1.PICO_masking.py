
#==============================================
# BEFORE RUNNING
#==============================================
'''

This is the code for making the "Masking image" of the FITS file for images taken by PICO (Ikeda et al. (2004)) 
The "Masking image" masks the 1) nearby stars and 2) Cosmic rays.


1. 
 - Input file:  
   '*.fits'         Preprocessed FITS file with the WCS implemented / each FITS file contain only one component (o-ray or e-ray)
   '*.mag.1'        IRAF Phot file containing target's center info.
                    See below (i.e.,2. What you need to run this code)
 
 - Output file:
   'mask_*.fits'    Masking image in FITS format
   
   

2. What you need to run this code. The following packages must be installed.
  - astropy (https://www.astropy.org/)
  - Astro-SCRAPPY (https://github.com/astropy/astroscrappy)
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
subpath  = 'The directory path where fits & mag.1 files are saved.'



#==============================================
# IMPORT PACKAGES AND DEFINE THE FUNCTION
#==============================================

import os
import glob
import matplotlib as mpl
import numpy as np
import pandas as pd
import warnings

import astroscrappy
from astropy.io import ascii, fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astropy.wcs import WCS
from astropy import units
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter('ignore', category=VerifyWarning)
mpl.rc('figure', max_open_warning = 0)
np.set_printoptions(threshold=1000)
pd.set_option('display.max_rows', None)

def pill_masking(image,x1,x2,y1,y2,g_star):
    
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
                    height =8
                elif  13 <= g_star[star] <= 15:
                    height=6
                elif  15 < g_star[star] < 18:
                    height=4
                elif  18 <= g_star[star]:
                    height=2
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
    return Masking_image






#==============================================
# BRING THE TARGET IMAGE TO BE MASKED
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


    
    
    
#==============================================
# MAKING THE MASKING IMAGE
#==============================================    
#The Masking image is masking the background star and the polarization mask area. 
#The position of the backgound stars are queried from astroquery.jplhorizons.Horizons and astroquery.gaia.Gaia



order = np.arange(0,len(file),8)
for z in order:
    SET = [file[z],file[z+1], file[z+2], file[z+3],file[z+4],file[z+5], file[z+6], file[z+7]]
    for i in range(0,8):
        RET = SET[i]  #Bring the fits file
        print(RET)

        #BRING THE IMAGE & ITS HEADER INFO    
        hdul = fits.open(RET)[0]
        header = hdul.header 
        image = hdul.data
        UT = header['DATE-OBS'] #Observation start, UT
        JD_str = Time(UT,format='isot').jd
        JD_end = JD_str + header['EXPTIME']/60/60/24

        #MAKE THE MASKED IMAGE
        Mask_image = np.zeros(np.shape(image))
        
              
        #MASKING THE BACKGROUND STARS
        #Bring the observer quantities from JPL Horizons
        #Querying the RA, DEC of target based on JD at exposure start
        obj = Horizons(id=Target_name,location=Observatory,epochs=JD_str)
        eph = obj.ephemerides()
        ra_str,dec_str = eph['RA'][0], eph['DEC'][0]   
        
        
        #Querying the RA, DEC of target based on JD at exposure end
        obj = Horizons(id=Target_name,location=Observatory,epochs=JD_end)
        eph = obj.ephemerides()
        ra_end,dec_end = eph['RA'][0], eph['DEC'][0]
        
        ra_tar = np.mean([ra_str,ra_end])
        dec_tar = np.mean([dec_str,dec_end])
        
        #Find the background stars's RA,DEC from Gaia
        coord = SkyCoord(ra=ra_tar, dec=dec_tar, unit=(units.degree, units.degree), frame='icrs')
        radi = units.Quantity(0.011, units.deg)
        result = Vizier.query_region(coord,
                        radius = radi,
                        catalog='I/345/gaia2')
        
        try:
            result[0]
        except IndexError:
            result = []
        else:
            result = result[0]
            
        RA_star = []
        DEC_star = []
        g_star = []
        if len(result) == 0:
            MASK = Mask_image
            print('No near-bay stars')
        else:
            for i in range(len(result)):
                if result['Gmag'][i]<18:
                    RA_star.append(result['RA_ICRS'][i])
                    DEC_star.append(result['DE_ICRS'][i])
                    g_star.append(result['Gmag'][i])
        
            #Convert the background stars's RA,DEC to pixel coordinate 
            Magfile = ascii.read(RET+'.mag.1')
            x,y = Magfile['XCENTER'][0]-1,Magfile['YCENTER'][0]-1 #Center of target     

        
            header['CRPIX1'], header['CRPIX2'] = x,y
            hdul.writeto(subpath+'/new_'+RET.split('/')[-1],overwrite=True)
            
            ### (X,Y) for the exposure start
            radis_star = []
            X_str = []
            Y_str = []          
            header['CRVAL1'], header['CRVAL2'] = ra_str,dec_str     
            w =WCS(header)    
            for i in range(len(RA_star)):
                x,y = w.wcs_world2pix(RA_star[i],DEC_star[i],0)
                X_str.append(float(x))
                Y_str.append(float(y))
                    
            ### (X,Y) for the exposure end
            X_end = []
            Y_end = []          
            header['CRVAL1'], header['CRVAL2'] = ra_end,dec_end     
            w =WCS(header)    

            for i in range(len(RA_star)):
                x,y = w.wcs_world2pix(RA_star[i],DEC_star[i],0)
                X_end.append(float(x))
                Y_end.append(float(y))   
                
            #Masking the stars    
            Masking_image_str = pill_masking(image,X_str,X_end,Y_str,Y_end,g_star)
            MASK = Mask_image  + Masking_image_str
            os.remove(subpath+'/new_'+RET.split('/')[-1])
            
        
        #MASK THE COSMIC-RAY
        gain = header['EGAIN']
        m_LA,cor_image = astroscrappy.detect_cosmics(image,
                                          sepmed = False,
                                          gain = gain,
                                          readnoise = RN,
                                           sigclip=5)
        tmLA = m_LA.astype(int)
        MASK[tmLA == 1 ] = np.nan        
        
        objpath = os.path.join(subpath,'mask_'+RET.split('/')[-1])
        fits.writeto(objpath,data = MASK,header = header,overwrite=True)
        print(objpath +' is created.')

