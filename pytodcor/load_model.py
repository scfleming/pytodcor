"""
.. module:: load_model
   :synopsis: Loads spectroscopic models from a supported source.
"""

import argparse
import logging

from pytodcor import supported_model_types
from pytodcor.read_model_bosz import read_model_bosz
from pytodcor.read_model_kurucz import read_model_kurucz

logger = logging.getLogger("load_model")

def load_model(model_type, teff, logg, metal):
    """
    Loads a set of stellar spectroscopic models from a supported library for a given
    effective temperature, surface gravity, and metallicity.
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
    """
    # Determine if `model_type` is a known, supported library.
    if model_type not in supported_model_types:
        logger.error("Requested type of model is not in the list of supported libraries. Must be "
                     "one of: (%s), received: %s", '; '.join(supported_model_types), model_type)
        raise ValueError("Requested type of model is not in the list of supported libraries."
                         " Must be one of: (" +
                         '; '.join(supported_model_types) +
                         f"), received: {model_type}")

    # Check if requested parameters are within the bounds of the models available.
    if model_type == "kurucz":
        pass
    elif model_type == "bosz":
        pass
    else:
        logger.error("Requested model parameters are outside the supported range for model "
                     "library: %s", model_type)
        raise ValueError("Requested model parameters are outside the supported range for model "
                          f"library: {model_type}.")

    # Identify the closest set of models to read in based on the input parameters.

    # Read in the model sets.

    # TO-DO: Linearly interpolate between the two bounding set of models if not
    # an exact match in the model grids.

def setup_args():
    parser = argparse.ArgumentParser(description="Load a model spectrum from a supported type.")
    parser.add_argument("model_type", action="store", type=str, help="[Required] Type of spectral "
                        "model to load.", choices=supported_model_types)
    parser.add_argument("teff", action="store", type=str, help="[Required] Effective temperature "
                        "of the set of models, in degrees Kelvin.")
    parser.add_argument("logg", action="store", type=str, help="[Required] Surface gravity of "
                        "the set of models, as log(g).")
    parser.add_argument("metal", action="store", type=str, help="[Required] Metallicity of "
                        "the set of models.")
    return parser.parse_args()

if __name__ == "__main__":
    INPUT_ARGS = setup_args()
    load_model(INPUT_ARGS.model_type, INPUT_ARGS.teff, INPUT_ARGS.logg, INPUT_ARGS.metal)
