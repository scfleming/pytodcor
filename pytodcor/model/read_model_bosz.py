"""
.. module:: read_model_bosz
   :synopsis: Reads a BOSZ model containing spectroscopic data.
"""

import logging
import numpy as np
import os
import pandas as pd
import gzip

from pytodcor import bosz_supported_resolutions

logger = logging.getLogger("read_model_bosz")

def read_model_bosz(bosz_root_dir, model_file):
    """
    Reads a BOSZ model file to extract spectroscopic information.

    :param bosz_root_dir: The path to the file containing the spectroscopic
                       model data to load.
    :type bosz_root_dir: str
    :param model_file: The name of the subdirectory and file containing the
                       spectroscopic model data to load.
    :type model_file: str
    :returns: tuple -- Spectroscopic data (wavelengths, continuum-subtracted fluxes) and object name based on the file.
    """
    if os.path.isfile(bosz_root_dir + model_file):
        # Read in the flux and continuum values.
        fl_df = pd.read_table(bosz_root_dir + model_file, delimiter=' ',
                               names=["fls", "conts"], dtype=np.float64)
        fls = np.asarray(fl_df["fls"])
        conts = np.asarray(fl_df["conts"])

        # Determine the resolution of the model from the file name.
        resolution = os.path.basename(model_file).split('_')[8][1:]
        if resolution not in bosz_supported_resolutions:
            logger.error("Resolution in model file not in list of supported resolutions: " + resolution)
            raise ValueError("Resolution in model file not in list of supported resolutions: " + resolution)

        # Check wavelength file exists and matches the resolution.
        bosz_resolution_file = "bosz2024_wave_r" + resolution + ".txt"
        if not os.path.isfile(bosz_root_dir + bosz_resolution_file):
            logger.error("Wavelength file not found: %s", bosz_root_dir +
                             bosz_resolution_file)
            raise IOError(f"Wavelength file not found: {bosz_root_dir +
                             bosz_resolution_file}")

        # Read in the wavelength file.
        wl_df = pd.read_table(bosz_root_dir + bosz_resolution_file,
                               names=["wls"], dtype=np.float64)
        if len(wl_df["wls"]) != len(fl_df["fls"]):
            logger.error("Wavelength and flux arrays not equal length: %s, %s",
                                 len(wl_df["wls"]), len(fl_df["fls"]))
            raise ValueError("Wavelength and flux arrays not equal length: " +
                                 str(len(wl_df['wls'])) + ", " +
                                 str(len(fl_df['fls'])))

        # Set target name based on model file (add logg as well for CK04).
        objname = os.path.basename(model_file).split('.txt.gz')[0]
    else:
        logger.error("File not found: %s", bosz_root_dir + model_file)
        raise IOError(f"File not found: {bosz_root_dir + model_file}")

    # Return the continuum-subtracted fluxes.
    return (objname, np.asarray(wl_df["wls"]),
                np.asarray(fl_df["fls"]/fl_df["conts"]))
