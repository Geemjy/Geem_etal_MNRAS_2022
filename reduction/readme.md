# Data Reduction

Here, we provide the code that we used to derive polarimetric results in Geem et al. (MNRAS 2022). The contents are

|Notebook, Script, Directory|Explanation|
|:----------------- |--------------- |
|[``1.FAPOL_maksing.ipynb``](1.FAPOL_maksing.ipynb)|The code for making the "Masking image" of FAPOL's data. |
|[``2.FAPOL_aperture_photometry.ipynb``](2.FAPOL_aperture_photometry.ipynb)|The code for the aperture photometry of FAPOL's data.|
|[``3.FAPOL_derive_qu..ipynb``](3.FAPOL_derive_qu.ipynb)|The code for deriving the $q$ and $u$ values of the target taken by FAPOL.|
|[``1.HONIR_masking.py``](1.HONIR_masking.py)|The code for making the "Masking image" of HONIR's data|
|[``2.HONIR_aper_photometry.py``](2.HONIR_aper_photometry.py)| The code for the aperture photometry of HONIR's data.|
|[``3.HONIR_derive_qu.py``](3.HONIR_derive_qu.py)|The code for deriving the $q$ and $u$ values of the target taken by HONIR.|
|[``1.WFGS2_masking.py``](1.WFGS2_masking.py)|The code for making the "Masking image" of WFGS2's data|
|[``2.WFGS2_aper_photometry.py``](2.WFGS2_aper_photometry.py)| The code for the aperture photometry of WFGS2's data.|
|[``3.WFGS2_derive_qu.py``](3.WFGS2_derive_qu.py)|The code for deriving the $q$ and $u$ values of the target taken by WFGS2.|
|[``1.PICO_masking.py``](1.PICO_masking.py)|The code for making the "Masking image" of PICO's data|
|[``2.PICO_aper_photometry.py``](2.PICO_aper_photometry.py)| The code for the aperture photometry of PICO's data.|
|[``3.PICO_derive_qu.py``](3.PICO_derive_qu.py)|The code for deriving the $q$ and $u$ values of the target taken by PICO.|

The polarimetric data (taken by FAPOL, HONIR and WFGS2) are available in Zenodo. 




## Requirements
Before running the script, the following packages must be installed. 

1. [astropy](https://www.astropy.org/) 
2. [Astro-SCRAPPY](https://github.com/astropy/astroscrappy) 
3. [Source Extraction and Photometry](https://sep.readthedocs.io/en/v1.1.x/index.html) 

    
## Contact

Created by Jooyeon Geem. - If you have any questions, please feel free to contact me (geem@astro.snu.ac.kr) !

The data reduction pipeline of MSI will continue to be developed in [@Geemjy](https://github.com/Geemjy) in the future.

``
