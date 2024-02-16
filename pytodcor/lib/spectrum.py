"""
.. module:: spectrum
   :synopsis: Defines the Spectrum class to store spectroscopic data.
"""

import logging
import numpy as np
from specutils.spectra import Spectrum1D
from specutils.manipulation import FluxConservingResampler

logger = logging.getLogger("spectrum")

class Spectrum:
    """
    This class defines a Spectrum object to store spectroscopic data and metadata necessary to
    perform cross-correlation and derive radial velocities.
    """
    def __init__(self, name="", air_or_vac="", obj_coord=None, juldate_utc=np.nan,
                 bjuldate_tdb=np.nan, tel_location="", exptime=np.nan,
                 teff=np.nan, logg=np.nan, metal=np.nan):
        """
        Class constructor.
        """
        # A spectrum is allowed to be in one or more "parts", loosely defined.  For example, each
        # detector in APOGEE can be considered a semi-independent "part" containing sets of
        # wavelengths and fluxes that can be cross-correlated indepdently, or individual orders of
        # an Echelle spectrum if the orders have not been stitched into a single spectrum.
        self.parts = []
        self.name = name
        if air_or_vac in ["", "air", "vacuum"]:
            self.air_or_vac = air_or_vac
        else:
            logger.error("air_or_vac argument given unsupported value, was given: %s", air_or_vac)
            raise ValueError("air_or_vac argument given unsupported value,"
                             f" was given: {air_or_vac}")
        self.obj_coord = obj_coord
        self.juldate_utc = juldate_utc
        self.bjuldate_tdb = bjuldate_tdb
        self.tel_location = tel_location
        self.exptime = exptime
        self.teff = teff
        self.logg = logg
        self.metal = metal

    def __add__(self, other):
        """
        Combines two spectra by combining their fluxes through addition. Wavelength grids must
        be identical. Does not combine any other metadata values. Fluxes are returned normalized
        by the maximum non-NaN value.
        """
        if len(self.parts) != len(other.parts):
            logger.error("Spectra must have the same number of parts to combine"
            " through addition.")
            raise ValueError("Spectra must have the same number of parts to combine"
            " through addition.")

        # New Spectrum object to be returned. Combine the names of the input sources together but
        # don't combine any other metadata.
        combined_spec = Spectrum(name=self.name + "," + other.name)

        for (ii, spart), opart in zip(enumerate(self.parts), other.parts):
            # Interpolate onto a common wavelength grid.
            fluxcon = FluxConservingResampler()
            opart_resampled = fluxcon(opart, spart.spectral_axis)
            # Add this part to the combined Spectrum class.
            combined_spec.add_spec_part(spart + opart_resampled)
            # Re-normalize the spectrum.
            combined_spec.parts[ii] /= np.nanmax(combined_spec.parts[ii].flux)
        return combined_spec

    def add_spec_part(self, spec):
        """
        Adds a new "part" of wavelengths and fluxes via a Spectrum1D object.

        :param spec: The one-dimensional spectrum for this part.
        :type spec: specutils.Spectrum1D
        """
        if isinstance(spec, Spectrum1D):
            self.parts.append(spec)
        else:
            logger.error("Attempt to add spectrum of unsupported type, must be a specutils"
                         " Spectrum1D object, was given type %s", str(type(spec)))
            raise ValueError("Attempt to add spectrum of unsupported type, must be a specutils"
                         f" Spectrum1D object, was given type {str(type(spec))}")
    