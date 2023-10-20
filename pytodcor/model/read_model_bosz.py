"""
.. module:: read_model_bosz
   :synopsis: Reads a BOSZ model containing spectroscopic data.
"""

import logging
import os
from astropy.io import fits

logger = logging.getLogger("read_model_bosz")

def read_model_bosz(model_file):
    """
    Reads a BOSZ model file to extract spectroscopic information.

    :param model_file: The full path and name of the file containing the spectroscopic
                       model data to load.
    :type model_file: str
    :returns: tuple -- Spectroscopic data and object name based on the file.
    """
    if os.path.isfile(model_file):
        with fits.open(model_file) as hdulist:
            dat1 = hdulist[1].data
        # Extract metadata from the primary header.
        # Set target name based on model file (add logg as well for CK04).
        objname = os.path.basename(model_file).split('.fits')[0]

        # Generate a Spectrum1D object.
        wls = dat1['wavelength']
        fls = dat1['specificintensity'] / dat1['continuum']
    else:
        logger.error("File not found: %s", model_file)
        raise IOError(f"File not found: {model_file}")
    return (objname, wls, fls)
