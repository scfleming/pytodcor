"""
.. module:: spectrum
   :synopsis: Defines the Spectrum class to store spectroscopic data.
"""

import logging
import numpy as np
from specutils.spectra import Spectrum1D

logger = logging.getLogger("spectrum")

class Spectrum:
    """
    This class defines a Spectrum object to store spectroscopic data and metadata necessary to
    perform cross-correlation and derive radial velocities.
    """
    def __init__(self, name="", air_or_vac="", obj_coord=None, juldate=np.nan, bjuldate=np.nan,
                 tel_lat=np.nan, tel_long=np.nan, exptime=np.nan):
        """
        Class constructor.
        """
        # A spectrum is allowed to be in one or more "parts", loosely defined.  For example, each
        # detector in APOGEE can be considered a semi-independent "part" containing sets of
        # wavelengths and fluxes that can be cross-correlated indepdently, or individual orders of
        # an Echelle spectrum if the orders have not been stitched into a single spectrum.
        self.parts = []
        self.name = name
        if air_or_vac in ["", "air", "vaccuum"]:
            self.air_or_vac = air_or_vac
        else:
            logger.error("air_or_vac argument given unsupported value, was given: %s", air_or_vac)
        self.obj_coord = obj_coord
        self.juldate = juldate
        self.bjuldate = bjuldate
        self.tel_lat = tel_lat
        self.tel_long = tel_long
        self.exptime = exptime

    def add_spec_part(self, spec):
        """
        Adds a new "part" of wavelengths and fluxes via a Spectrum1D object.

        :param spec: The one-dimensional spectrum for this part.
        :type wls: specutils.Spectrum1D
        """
        if isinstance(spec, Spectrum1D):
            self.parts.append(spec)
        else:
            logger.error("Attempt to add spectrum of unsupported type, must be a specutils"
                         " Spectrum1D object, was given type %s", str(type(spec)))
