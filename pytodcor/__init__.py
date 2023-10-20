"""
.. module:: __init__
   :synopsis: Initialization module.
"""

import logging
import logging.config
import os

# Create the Logger
loggers = logging.getLogger()
loggers.setLevel(logging.DEBUG)

# Create the Handler for logging data to a file
__LOGFILE_NAME = "pytodcor.log"
if os.path.isfile(__LOGFILE_NAME):
    os.remove(__LOGFILE_NAME)
logger_handler = logging.FileHandler(filename=__LOGFILE_NAME)
logger_handler.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
loggers.addHandler(logger_handler)
loggers.info('Completed configuring logger()!')

__version__ = "0.1"

# Supported types of spectroscopic data.
#    "apogee" = APOGEE apVisit spectra from the Sloan Digital Sky Survey in FITS format.
#    "arc35" = Calibrated, one-dimensional ARC 3.5m Echelle spectra in FITS format.
supported_spec_types = ["apogee", "arc35"]

# Supported types of spectral models.
#    "kurucz" = Castelli & Kurucz 2004 models
#    "bosz" = Bohlin et al. 2017 ATLAS9 models
supported_models = {"types":["bosz", "kurucz"],
                    "lookup_files":{"bosz":"bosz_model_files_and_parmas.txt",
                                     "kurucz":"kurucz_model_files_and_params.txt"}
                    }
