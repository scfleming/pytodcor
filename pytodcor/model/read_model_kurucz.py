"""
.. module:: read_model_kurucz
   :synopsis: Reads a Castelli-Kurucz model containing spectroscopic data.
"""

import logging
import os
from astropy.io import fits

logger = logging.getLogger("read_model_kurucz")

def read_model_kurucz(model_file, model_logg):
    """
    Reads a Castelli-Kurucz model file to extract spectroscopic information.

    :param model_file: The full path and name of the file containing the spectroscopic
                       model data to load.
    :type model_file: str
    :param model_logg: The log(g) surface gravity of the model to retrieve.
    :type model_logg: float
    :returns: tuple -- Spectroscopic data and object name based on the file.
    """
    if os.path.isfile(model_file):
        with fits.open(model_file) as hdulist:
            dat1 = hdulist[1].data
        # Extract metadata from the primary header.
        # Set target name based on model file (add logg as well for CK04).
        objname = os.path.basename(model_file).split('.fits')[0] + "_g" + str(model_logg)

        # Generate a Spectrum1D object.
        wls = dat1['wavelength']
        # The spectral for different surface gravities are stored in columns defined by
        # the value of log(g) * 10, e.g., log(g) = 0.5 is stored in column "g05".
        logg_colname = 'g' + str(int(model_logg*10))
        fls = dat1[logg_colname]
    else:
        logger.error("File not found: %s", model_file)
        raise IOError(f"File not found: {model_file}")
    return (objname, wls, fls)
