"""
.. module:: load_spectrum
   :synopsis: Loads spectroscopic data from a supported source.
"""

import argparse
import logging

from pytodcor import supported_spec_types
from pytodcor.read_spec_apogee import read_spec_apogee
from pytodcor.read_spec_arc35 import read_spec_arc35

logger = logging.getLogger("load_spectrum")

def load_spectrum(spec_type, spec_file):
    """
    Reads a file containing spectroscopic data to do cross-correlation on.

    :param spec_type: The type of spectrum to load. The data format and creation of the Spectrum
                      object depends on the source of the spectrum. Consult the documentation for
                      more information on what types of spectroscopic data products are supported.
    :type spec_type: str

    :param spec_file: The full path and name of the file containing the spectroscopic data to load.
    :type spec_file: str

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
        this_spec = read_spec_arc35(spec_file)

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
    return parser.parse_args()

if __name__ == "__main__":
    INPUT_ARGS = setup_args()
    load_spectrum(INPUT_ARGS.spec_type, INPUT_ARGS.spec_file)
