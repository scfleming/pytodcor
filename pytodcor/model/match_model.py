"""
.. module:: match_model
   :synopsis: Identifies the closest spectroscopic models from a supported source given
              requested stellar parameters.
"""
import logging
import os
import numpy as np
from pandas import read_csv

from pytodcor import supported_models

logger = logging.getLogger("match_model")

def _find_bounding_vals(df, teff, logg, metal):
    """
    Given a pandas data frame finds the location of the closest lower and upper
    indices to a requested value.
    Credit to https://stackoverflow.com/users/9726179/ivo-merchiers per
    https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
    with slight modifications here to extend to multiple columns.
    """
    exactmatch = df[(df["teff"] == teff) &
                    (df["logg"] == logg) &
                    (df["metallicity"] == metal)]
    if not exactmatch.empty:
        return df.iloc[exactmatch.index]
    # These return the index of the first instance of the lower and upper bounding
    # values for each parameter.
    lower_ind = df[(df["teff"] <= teff) & (df["logg"] <= logg) &
                    (df["metallicity"] <= metal)].idxmax(numeric_only=True)
    upper_ind = df[(df["teff"] >= teff) & (df["logg"] >= logg) &
                    (df["metallicity"] >= metal)].idxmin(numeric_only=True)
    # Now find the index in the data frame that matches all the parameters found
    # from the index search.
    metal_ind = lower_ind.index.get_loc("metallicity")
    metal_low = df[lower_ind.index[metal_ind]][lower_ind[metal_ind]]
    metal_high = df[lower_ind.index[metal_ind]][upper_ind[metal_ind]]

    teff_ind = lower_ind.index.get_loc("teff")
    teff_low = df[lower_ind.index[teff_ind]][lower_ind[teff_ind]]
    teff_high = df[lower_ind.index[teff_ind]][upper_ind[teff_ind]]

    logg_ind = lower_ind.index.get_loc("logg")
    logg_low = df[lower_ind.index[logg_ind]][lower_ind[logg_ind]]
    logg_high = df[lower_ind.index[logg_ind]][upper_ind[logg_ind]]

    # For a three-dimensional parameter space, return the grid locations of all
    # combaintions of low and high values for the three parameters to allow for
    # optimal interpolation in the future.  If one parameter happens to lie
    # on an exact grid location, some of these may be identical indices.
    return_ind_01 = df[(df["metallicity"] == metal_low) &
                        (df["teff"] == teff_low) & (df["logg"] == logg_low)].index
    return_ind_02 = df[(df["metallicity"] == metal_low) &
                        (df["teff"] == teff_high) & (df["logg"] == logg_low)].index
    return_ind_03 = df[(df["metallicity"] == metal_low) &
                        (df["teff"] == teff_low) & (df["logg"] == logg_high)].index
    return_ind_04 = df[(df["metallicity"] == metal_low) &
                        (df["teff"] == teff_high) & (df["logg"] == logg_high)].index
    return_ind_05 = df[(df["metallicity"] == metal_high) &
                        (df["teff"] == teff_low) & (df["logg"] == logg_low)].index
    return_ind_06 = df[(df["metallicity"] == metal_high) &
                        (df["teff"] == teff_high) & (df["logg"] == logg_low)].index
    return_ind_07 = df[(df["metallicity"] == metal_high) &
                        (df["teff"] == teff_low) & (df["logg"] == logg_high)].index
    return_ind_08 = df[(df["metallicity"] == metal_high) &
                        (df["teff"] == teff_high) & (df["logg"] == logg_high)].index
    return_inds = list(np.concatenate([return_ind_01.values, return_ind_02.values,
                                       return_ind_03.values, return_ind_04.values,
                                       return_ind_05.values, return_ind_06.values,
                                       return_ind_07.values, return_ind_08.values]))
    return df.iloc[return_inds]

def match_model(model_type, teff, logg, metal, lookup_dir):
    """
    Identifies the closest models based on requested stellar parameters.
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
    :param lookup_dir: The root directory containing the lookup table for the models.
    :type lookup_dir: str
    :returns: list -- The closest models to the requested stellar parameters. The length
                      of the return can be a one-element or an eight-element list, depending
                      on if there's an exact match or a need to provide a bounding box.
    """
    # Read in the look-up table for this model.

    if not os.path.isdir(lookup_dir):
        logger.error("Model directory not found: %s", lookup_dir)
        raise IOError(f"Model directory not found: {lookup_dir}")
    lookup_table_file = lookup_dir + supported_models["lookup_files"][model_type]
    if not os.path.isfile(lookup_table_file):
        logger.error("Model lookup table not found: %s", lookup_table_file)
        raise IOError(f"Model lookup table not found: {lookup_table_file}")
    df_models = read_csv(lookup_table_file, sep=',', dtype={"file":str, "metallicity":float,
                                                             "teff":float, "logg":float})

    # Locate the closest model match(es) for the requested stellar parameters.
    closest_models = _find_bounding_vals(df_models, teff, logg, metal)
    return closest_models
