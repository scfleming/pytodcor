"""
.. module:: load_spectrum
   :synopsis: Loads spectroscopic data from a supported source.
"""

import argparse
import logging

from pytodcor import supported_spec_types
from pytodcor.obs.read_spec_apogee import read_spec_apogee
from pytodcor.obs.read_spec_arc35 import read_spec_arc35

logger = logging.getLogger("load_spectrum")

def load_spectrum(spec_type, spec_file, wl_min=None, wl_max=None):
    """
    Reads a file containing spectroscopic data to do cross-correlation on.

    :param spec_type: The type of spectrum to load. The data format and creation of the Spectrum
                      object depends on the source of the spectrum. Consult the documentation for
                      more information on what types of spectroscopic data products are supported.
    :type spec_type: str

    :param spec_file: The full path and name of the file containing the spectroscopic data to load.
    :type spec_file: str
    :param wl_min: The minimum wavelength, in Angstroms, of a subset of the spectrum
                    to return if the full spectrum isn't requested.
    :type wl_min: float
    :param wl_max: The maximum wavelength, in Angstroms, of a subset of the spectrum
                    to return if the full spectrum isn't requested.
    :type wl_max: float
    :returns: pytodcor.Spectrum -- The spectroscopic data contained in the input file.
    """

    # Sanitize input type of spectrum.
    spec_type = spec_type.lower().strip()

    # Determine if `spec_type` is a known, supported source.
    if spec_type not in supported_spec_types:
        logger.error("Requested type of spectrum is not in the list of supported types. Must be "
                     "one of: (%s), received: %s", '; '.join(supported_spec_types), spec_type)
        raise ValueError("Requested type of spectrum is not in the list of supported types."
                         " Must be one of: (" +
                         '; '.join(supported_spec_types) +
                         f"), received: {spec_type}")

    # Call the function to load the spectroscopic data file.
    if spec_type == "apogee":
        logger.info("Reading APOGEE spectroscopic data file: %s", spec_file)
        this_spec = read_spec_apogee(spec_file)
    elif spec_type == "arc35":
        logger.info("Reading ARC 3.5m spectroscopic data file: %s", spec_file)
        this_spec = read_spec_arc35(spec_file, wl_min, wl_max)

    return this_spec

def setup_args():
    """
    If running from the command-line, the following is executed.
    """
    parser = argparse.ArgumentParser(description="Load a Spectrum from an input file.")
    parser.add_argument("spec_type", action="store", type=str, help="[Required] Type of spectral "
                        "data to load.", choices=supported_spec_types)
    parser.add_argument("spec_file", action="store", type=str, help="[Required] Full path and "
                        "name of the input file.")
    parser.add_argument("--wl_min", action="store", type=float, help="Limit read-in spectra"
                        " to only wavelengths greater than this value, specified in Angstroms.",
                          default=None)
    parser.add_argument("--wl_max", action="store", type=float, help="Limit read-in spectra"
                        " to only wavelengths less than this value, specified in Angstroms.",
                          default=None)
    return parser.parse_args()

if __name__ == "__main__":
    INPUT_ARGS = setup_args()
    load_spectrum(INPUT_ARGS.spec_type, INPUT_ARGS.spec_file, INPUT_ARGS.wl_min, INPUT_ARGS.wl_max)
