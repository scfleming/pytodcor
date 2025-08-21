"""
.. module:: read_model_bosz
   :synopsis: Reads a BOSZ model containing spectroscopic data.
"""

import logging
import os
from astropy.io import fits
import numpy as np
import gzip


logger = logging.getLogger("read_model_bosz")

def read_model_bosz(model_file):
    """
    Reads a BOSZ model file to extract spectroscopic information.

    :param model_file: The full path and name of the file containing the spectroscopic
                       model data to load.
    :type model_file: str
    :returns: tuple -- Spectroscopic data and object name based on the file.
    """
    #print(f"Trying to open file: {model_file}") something I had to debug
    if os.path.isfile(model_file):
        #You cannot use fits stuff here, it's a txt.gz. Not sure how else to extract stuff for the hdulist
        #"fluxes and continuum values are stored at these lower instrumental broadenings
        with fits.open(model_file) as hdulist:
            dat1 = hdulist[1].data
        # Extract metadata from the primary header.
        # Set target name based on model file (add logg as well for CK04).
        objname = os.path.basename(model_file).split('.fits')[0]

        # Generate a Spectrum object.
        wls = dat1['wavelength']
        fls = dat1['specificintensity'] / dat1['continuum']
    else:
        logger.error("File not found: %s", model_file)
        raise IOError(f"File not found: {model_file}")
    ''' The following is just what I tried, should the way to open this txt.gz,
     not sure how to extract the header information usually found in a fit file. 
    '''
    '''if os.path.isfile(model_file):
            with gzip.open(model_file, 'rt') as f:
                wls = []
                fls = []
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            w, f = map(float, line.strip().split())
                            wls.append(w)
                            fls.append(f)
                        except ValueError:
                            print(f"Skipping malformed line: {line.strip()}")
                            continue
                wls = np.array(wls, dtype=np.float64)
                fls = np.array(fls, dtype=np.float64)
                objname = os.path.basename(model_file).split('.txt.gz')[0]
                return (objname, wls, fls)
        else:
            logger.error("File not found: %s", model_file)
            raise IOError(f"File not found: {model_file}")'''
    return (objname, wls, fls)


