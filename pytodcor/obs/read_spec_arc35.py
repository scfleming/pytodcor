"""
.. module:: read_spec_arc35
   :synopsis: Reads an APO 3.5m ARC file containing spectroscopic data.
"""

import logging
import os
import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coord
from astropy.time import Time
import astropy.wcs as apwcs
from specutils.spectra import Spectrum1D

from pytodcor.lib.spectrum import Spectrum

logger = logging.getLogger("read_spec_arc35")

def read_spec_arc35(spec_file, wl_min=None, wl_max=None):
    """
    Reads an APO 3.5m ARC file to extract spectroscopic information.

    :param spec_file: The full path and name of the file containing the spectroscopic data to load.
    :type spec_file: str
    :param wl_min: The minimum wavelength, in Angstroms, of a subset of the spectrum
                    to return if the full spectrum isn't requested.
    :type wl_min: float
    :param wl_max: The maximum wavelength, in Angstroms, of a subset of the spectrum
                    to return if the full spectrum isn't requested.
    :type wl_max: float
    :returns: dict -- Spectroscopic data and metadata from the file.
    """
    if os.path.isfile(spec_file):
        with fits.open(spec_file) as hdulist:
            hdr0 = hdulist[0].header
            dat0 = hdulist[0].data
        # Extract metadata from the primary header.

        # Target name from the header.
        objname = hdr0['OBJNAME']

        # Target coordinates.
        equinox = hdr0['EQUINOX']
        if equinox == 2000.0:
            this_equinox = 'J2000'
        else:
            logger.error("Did not find a supported value in the EQUINOX header keyword,"
                         " found value = %s", str(equinox))
            raise ValueError("Did not find a supported value in the EQUINOX header keyword,"
                         f" found value = {str(equinox)}")
        this_coord = coord.SkyCoord(ra=hdr0['RA'], dec=hdr0['DEC'], unit=(u.hourangle, u.deg),
                               frame=hdr0['RADECSYS'].lower(), equinox=this_equinox)

        # Total exposure time in seconds, to calculate time at mid-point of integration.
        exptime = hdr0['EXPTIME']
        this_ut_mid = Time(hdr0['UT1'], format='isot', scale='ut1',
                           location=coord.EarthLocation.of_site('greenwich')) + (exptime/2.)*u.s
        # Barycentric correction to the time in days.
        ltt_barycorr = this_ut_mid.light_travel_time(this_coord)
        this_tdb_bjd_mid = this_ut_mid.tdb.jd*u.day + ltt_barycorr

        # Generate a Spectrum1D object.
        this_wcs = apwcs.WCS(header={'CDELT1': hdr0['CDELT1'], 'CRVAL1': hdr0['CRVAL1'],
                                      'CUNIT1': 'Angstrom', 'CTYPE1': hdr0['CTYPE1'],
                                      'CRPIX1': hdr0['CRPIX1']})
        these_wls = this_wcs.pixel_to_world(range(len(dat0)))
        these_fls = dat0*u.dimensionless_unscaled

        # If requested, extract only a subset based on the wavelength range.
        keep_indices = np.where(
            (these_wls >= wl_min*u.angstrom if wl_min else np.isfinite(these_wls)) &
            (these_wls <= wl_max*u.angstrom if wl_max else np.isfinite(these_wls)))[0]
        if len(keep_indices) > 0:
            these_wls = these_wls[keep_indices]
            these_fls = these_fls[keep_indices]
        else:
            logger.error("No model spectra contained within requested wavelength range: " +
                        "%s <= wavelength <= %s", str(wl_min), str(wl_max))
            raise ValueError("No model spectra contained within requested wavelength range: " +
                        f"{str(wl_min)} <= wavelength <= {str(wl_max)}")

        # Construct the Spectrum object.
        spec = Spectrum1D(flux=these_fls, spectral_axis=these_wls)
        this_spec = Spectrum(name=objname, air_or_vac="air", obj_coord=this_coord,
                             juldate_utc=this_ut_mid.utc.jd, bjuldate_tdb=this_tdb_bjd_mid,
                             tel_location="apo", exptime=exptime)
        this_spec.add_spec_part(spec)
    else:
        logger.error("File not found: %s", spec_file)
        raise IOError(f"File not found: {spec_file}")
    return this_spec
