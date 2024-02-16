"""
.. module:: load_model
   :synopsis: Loads spectroscopic models from a supported source.
"""

import argparse
import logging

from astropy import units as u
import numpy as np
from specutils.spectra import Spectrum1D

from pytodcor import supported_models
from pytodcor.lib.spectrum import Spectrum
from pytodcor.model.match_model import match_model
from pytodcor.model.read_model_bosz import read_model_bosz
from pytodcor.model.read_model_kurucz import read_model_kurucz

logger = logging.getLogger("load_model")

def _check_bounds(model_type, teff, logg, metal):
    """
    Verifies requested parameters are within the bounds of the model library being requested.
    :param model_type: The type of model to load. The data format and set of models
                       that get loaded depends on the library being used.
    
    :type model_type: str
    :param teff: The effective temperature to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    :type teff: float
    :param logg: The surface gravity to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    :type logg: float
    :param metal: The metallicity to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    """
    if model_type == "kurucz":
        teff_min_val = 3500.
        teff_max_val = 50000.
        if teff < teff_min_val or teff > teff_max_val:
            logger.error("Temperature of model outside bounds supported by Kurucz library. Must be "
                         "> %s and < %s.", teff_min_val, teff_max_val)
            raise ValueError("Temperature of model outside bounds supported by Kurucz library."
                             f" Must be > {teff_min_val} and < {teff_max_val}.")
        logg_min_val = 0.
        logg_max_val = 5.
        if logg < logg_min_val or logg > logg_max_val:
            logger.error("Surface gravity of model outside bounds supported by Kurucz library."
                         " Must be > %s and < %s.", logg_min_val, logg_max_val)
            raise ValueError("Surface gravity of model outside bounds supported by Kurucz"
                             f" library. Must be > {logg_min_val} and < {logg_max_val}.")
        metal_min_val = -2.5
        metal_max_val = 0.5
        if metal < metal_min_val or metal > metal_max_val:
            logger.error("Metallicity of model outside bounds supported by Kurucz library."
                         " Must be > %s and < %s.", metal_min_val, metal_max_val)
            raise ValueError("Metallicity of model outside bounds supported by Kurucz"
                             f" library. Must be > {metal_min_val} and < {metal_max_val}.")
    elif model_type == "bosz":
        teff_min_val = 3500.
        teff_max_val = 10000.
        if teff < teff_min_val or teff > teff_max_val:
            logger.error("Temperature of model outside bounds supported by BOSZ library. Must be "
                         "> %s and < %s.", teff_min_val, teff_max_val)
            raise ValueError("Temperature of model outside bounds supported by BOSZ library."
                             f" Must be > {teff_min_val} and < {teff_max_val}.")
        if teff < 6000.:
            logg_min_val = 0.
            logg_max_val = 5.
        elif teff < 8000.:
            logg_min_val = 1.
            logg_max_val = 5.
        elif teff < 12000.:
            logg_min_val = 2.
            logg_max_val = 5.
        if logg < logg_min_val or logg > logg_max_val:
            logger.error("Surface gravity of model outside bounds supported by BOSZ library."
                         " Must be > %s and < %s.", logg_min_val, logg_max_val)
            raise ValueError("Surface gravity of model outside bounds supported by BOSZ"
                             f" library. Must be > {logg_min_val} and < {logg_max_val}.")
        metal_min_val = -5.0
        metal_max_val = 1.5
        if metal < metal_min_val or metal > metal_max_val:
            logger.error("Metallicity of model outside bounds supported by BOSZ library."
                         " Must be > %s and < %s.", metal_min_val, metal_max_val)
            raise ValueError("Metallicity of model outside bounds supported by BOSZ"
                             f" library. Must be > {metal_min_val} and < {metal_max_val}.")

def load_model(model_type, teff, logg, metal, ck04_root_dir="templates/ck04/",
                bosz_root_dir="templates/bosz/", model_wl_min=None, model_wl_max=None):
    """
    Loads a stellar spectroscopic model from a supported library for a given
    effective temperature, surface gravity, and metallicity. Will interpolate
    between model grids as needed.

    :param model_type: The type of model to load. The data format and set of models
                       that get loaded depends on the library being used.
    :type model_type: str
    :param teff: The effective temperature to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    :type teff: float
    :param logg: The surface gravity to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    :type logg: float
    :param metal: The metallicity to retrieve a model set for. The closest
                 match will be retrieved, as long as it is not outside the range of
                 the library being used.
    :type metal: float
    :param ck04_root_dir: The root directory containing the Castelli-Kurucz 2004 models.
    :type ck04_root_dir: str
    :param bosz_root_dir: The root directory containing the BOSZ models.
    :type bosz_root_dir: str
    :param model_wl_min: The minimum wavelength, in Angstroms, of a subset of the spectrum
                         to return if the full model spectrum isn't requested.
    :type model_wl_min: float
    :param model_wl_max: The maximum wavelength, in Angstroms, of a subset of the spectrum
                         to return if the full model spectrum isn't requested.
    :type model_wl_max: float
    :returns: pytodcor.Spectrum -- The spectrum derived from the models.
    """
    # Determine if `model_type` is a known, supported library.
    if model_type not in supported_models["types"]:
        logger.error("Requested type of model is not in the list of supported libraries. Must be "
                     "one of: (%s), received: %s", '; '.join(supported_models["types"]), model_type)
        raise ValueError("Requested type of model is not in the list of supported libraries."
                         " Must be one of: (" +
                         '; '.join(supported_models["types"]) +
                         f"), received: {model_type}")

    # Check if requested parameters are within the bounds of the models available.
    _check_bounds(model_type, teff, logg, metal)

    # Identify the closest set of models to read in based on the input parameters.
    if model_type == "ck04":
        lookup_dir = ck04_root_dir
    elif model_type == "bosz":
        lookup_dir = bosz_root_dir
    models_to_read = match_model(model_type, teff, logg, metal, lookup_dir)

    # Read in the model sets.
    model_set = []
    for iii in range(len(models_to_read)):
        if model_type == "kurucz":
            # Read in the models. Must pass in log(g) value since all gravities are grouped
            # into one file.
            this_airorvac = "vacuum"
            (objname, these_wls, these_fls) = read_model_kurucz(ck04_root_dir +
                                                                 models_to_read['file'].iloc[iii],
                                                                 models_to_read["logg"].iloc[iii])
        elif model_type == "bosz":
            # Read in the models.
            this_airorvac = "vacuum"
            (objname, these_wls, these_fls) = read_model_bosz(bosz_root_dir +
                                                               models_to_read['file'].iloc[iii])

        # If requested, extract only a subset based on the wavelength range.
        keep_indices = np.where(
            (these_wls >= model_wl_min if model_wl_min else np.isfinite(these_wls)) &
            (these_wls <= model_wl_max if model_wl_max else np.isfinite(these_wls)))[0]
        if len(keep_indices) > 0:
            these_wls = these_wls[keep_indices]
            these_fls = these_fls[keep_indices]
        else:
            logger.error("No model spectra contained within requested wavelength range: " +
                        "%s <= wavelength <= %s", str(model_wl_min), str(model_wl_max))
            raise ValueError("No model spectra contained within requested wavelength range: " +
                        f"{str(model_wl_min)} <= wavelength <= {str(model_wl_max)}")

        # Noramlize the fluxes since only flux-normalized spectra are needed for cross-correlation.
        these_fls = these_fls / np.nanmax(these_fls)
        spec = Spectrum1D(flux=these_fls*u.dimensionless_unscaled,
                           spectral_axis=these_wls*u.angstrom)

        # Construct the Spectrum object.
        this_spec = Spectrum(name=objname, air_or_vac=this_airorvac)
        this_spec.add_spec_part(spec)
        model_set.append(this_spec)

    # TO-DO: Linearly interpolate between the two bounding set of models if not
    # an exact match in the model grids.

    # Return the model spectrum. Until interpolation is achieved, just return the
    # first one in the matched set.
    return model_set[0]

def setup_args():
    """
    If running from the command-line, the following is executed.
    """
    parser = argparse.ArgumentParser(description="Load a model spectrum from a supported type.")
    parser.add_argument("model_type", action="store", type=str, help="[Required] Type of spectral "
                        "model to load.", choices=supported_models["types"])
    parser.add_argument("teff", action="store", type=float, help="[Required] Effective temperature "
                        "of the set of models, in degrees Kelvin.")
    parser.add_argument("logg", action="store", type=float, help="[Required] Surface gravity of "
                        "the set of models, as log(g).")
    parser.add_argument("metal", action="store", type=float, help="[Required] Metallicity of "
                        "the set of models.")
    parser.add_argument("--ck04_root_dir", action="store", type=str, help="Root directory"
                        " containing the Castelli-Kurucz 2004 model sets.",
                          default="templates/ck04/")
    parser.add_argument("--bosz_root_dir", action="store", type=str, help="Root directory"
                        " containing the BOSZ model sets.", default="templates/bosz/")
    parser.add_argument("--model_wl_min", action="store", type=float, help="Limit model spectra"
                        " to only wavelengths greater than this value, specified in Angstroms.",
                          default=None)
    parser.add_argument("--model_wl_max", action="store", type=float, help="Limit model spectra"
                        " to only wavelengths less than this value, specified in Angstroms.",
                          default=None)
    return parser.parse_args()

if __name__ == "__main__":
    INPUT_ARGS = setup_args()
    load_model(INPUT_ARGS.model_type, INPUT_ARGS.teff, INPUT_ARGS.logg, INPUT_ARGS.metal,
                INPUT_ARGS.ck04_root_dir, INPUT_ARGS.bosz_root_dir, INPUT_ARGS.model_wl_min,
                INPUT_ARGS.model_wl_max)
