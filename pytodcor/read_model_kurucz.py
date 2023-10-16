"""
.. module:: read_model_kurucz
   :synopsis: Reads a Castelli-Kurucz model containing spectroscopic data.
"""

import logging
import os
from astropy import units as u
from astropy.io import fits
import numpy as np
from specutils.spectra import Spectrum1D
from pytodcor.spectrum import Spectrum

logger = logging.getLogger("read_model_kurucz")

def read_model_kurucz(model_file, model_logg, model_wl_min, model_wl_max):
    """
    Reads a Castelli-Kurucz model file to extract spectroscopic information.

    :param model_file: The full path and name of the file containing the spectroscopic
                       model data to load.
    :type model_file: str
    :param model_logg: The log(g) surface gravity of the model to retrieve.
    :type model_logg: float
    :param model_wl_min: The minimum wavelength, in Angstroms, of a subset of the spectrum
                         to return if the full model spectrum isn't requested.
    :type model_wl_min: float
    :param model_wl_max: The maximum wavelength, in Angstroms, of a subset of the spectrum
                         to return if the full model spectrum isn't requested.
    :type model_wl_max: float
    :returns: dict -- Spectroscopic data and metadata from the file.
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
        # If requested, extract only a subset based on the wavelength range.
        keep_indices = np.where((wls >= model_wl_min if model_wl_min else np.isfinite(wls)) &
                                (wls <= model_wl_max if model_wl_max else np.isfinite(wls)))[0]
        if len(keep_indices) > 0:
            wls = wls[keep_indices]
            fls = fls[keep_indices]
        else:
            logger.error("No model spectra contained within requested wavelength range: " +
                        "%s <= wavelength <= %s", str(model_wl_min), str(model_wl_max))
            raise ValueError("No model spectra contained within requested wavelength range: " +
                        f"{str(model_wl_min)} <= wavelength <= {str(model_wl_max)}")
        # Noramlize the fluxes since only flux-normalized spectra are needed for cross-correlation.
        fls = fls / np.nanmax(fls)
        spec = Spectrum1D(flux=fls*u.dimensionless_unscaled, spectral_axis=wls*u.angstrom)

        # Construct the Spectrum object.
        this_spec = Spectrum(name=objname, air_or_vac="vacuum")
        this_spec.add_spec_part(spec)
    else:
        logger.error("File not found: %s", model_file)
        raise IOError(f"File not found: {model_file}")
    return this_spec
