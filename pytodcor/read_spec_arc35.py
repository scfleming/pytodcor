"""
.. module:: read_spec_arc35
   :synopsis: Reads an APO 3.5m ARC file containing spectroscopic data.
"""

import logging
import os
from astropy import units as u
from astropy.io import fits
from astropy import coordinates as coord
from astropy.time import Time
import astropy.wcs as apwcs
from specutils.spectra import Spectrum1D
from pytodcor.spectrum import Spectrum

logger = logging.getLogger("read_spec_arc35")

def read_spec_arc35(spec_file):
    """
    Reads an APO 3.5m ARC file to extract spectroscopic information.

    :param spec_file: The full path and name of the file containing the spectroscopic data to load.
    :type spec_file: str

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
        spec = Spectrum1D(flux=dat0*u.dimensionless_unscaled, wcs=this_wcs)

        # Construct the Spectrum object.
        this_spec = Spectrum(name=objname, air_or_vac="air", obj_coord=this_coord,
                             juldate_utc=this_ut_mid.utc.jd, bjuldate_tdb=this_tdb_bjd_mid,
                             tel_location="apo", exptime=exptime)
        this_spec.add_spec_part(spec)
    else:
        logger.error("File not found: %s", spec_file)
        raise IOError(f"File not found: {spec_file}")
    return this_spec
