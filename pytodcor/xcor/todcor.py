"""
.. module:: todcor
   :synopsis: Performs two-dimensional cross-correlation following Zucker & Mazeh 1994,
   ApJ, 420, 806.
"""

import logging
import math
import time
from astropy import units as u
import numpy as np
from scipy.constants import c as speedoflight
from specutils.analysis import template_logwl_resample
from specutils.manipulation import FluxConservingResampler
from pytodcor.xcor.xcor1d import xcor1d

logger = logging.getLogger("todcor")

def _find_rel_shift(lag_i, lag_j):
    """
    Determines the correct index to access in a cross-correlation lag array given a shift
    applied to each of the two signals. A shift of "-s1" combined with a shift of "+s2"
    is equivalent to holding signal_1 constant and applying a shift of "+s2 - -s1" to signal_2.
    Similarly a shift of "+s1" and a shift of "-s2" is equivalent to holding signal_1
    constant and applying a shift of "-s2 - s1" to signal_2. A shift to both signals in the
    same direction is similarly equivalent to keeping signal_1 constant and shifting signal_2
    by the difference between "s2 - s1" or "-s2 - -s1". The function itself is trivial, but
    working through the scenarios required some forethought so making it a separate function
    allows for proper testing as needed in case the logic behind indexing into the
    cross-correlation results is incorrect.
    """
    return lag_j - lag_i

def todcor(obs_spec, model_1, model_2, n_pix_shifts, fixed_alpha=None, vel_range=None):
    """
    Performs two-dimensional cross-correlation given an observed spectrum and two model spectra to
    use as templates. Spectral wavelengths are assumed to be in the same units between the observed
    and model spectra. Spectral fluxes are assumed to be normalized in all cases.

    :param obs_spec: The observed spectrum.
    :type obs_spec: pytodcor.Spectrum
    :param model_1: The first model template.
    :type model_1: pytodcor.Spectrum
    :param model_2: The second model template.
    :type model_2: pytodcor.Spectrum
    :param n_pix_shifts: The number of pixels to shift the templates in one direction, will
                         calculate +/- this amount and no shift, thus resulting in
                         2*n_pix_shift+1 shifts.
    :type n_pix_shifts: int
    :param fixed_alpha: The flux ratio of the second template compared to the first. If not
                        fixed it will be calculated following the TODCOR algorithm.
    :type fixed_alpha: float
    :param vel_range: The minimun and maximum radial velocities to consider when shifting the
                    templates, in km/s. If not set, a default of [-1000, 1000] km/s is used.
    :type vel_range: list
    :returns: tuple -- A tuple containing the array of pixel shifts (applied equally to both
                       templates), the pixel-to-velocity scaling factor (multiply by this to
                       convert pixel shifts to velocity), a two-dimensional array of TODCOR
                       correlation values, and a two-dimensional array of TODCOR scaling
                       ratios between the two templates.
    """

    # Set a velocity range limit if none is provided.
    if not vel_range:
        vel_range = [-1000., 1000.]

    # Resample the obseerved spectrum, model_1 and model_2 spectra to the same log-lambda scale.
    # Use specutils.analysis.template_logwl_resample() method,
    # https://specutils.readthedocs.io/en/stable/api/specutils.analysis.template_logwl_resample.html
    logger.info("Resampling observed spectrum and first template to a common log-lambda wavelength"
                " scale...")
    print("Resampling observed spectrum and first template to a common log-lambda wavelength"
          " scale...")
    tstart = time.process_time()
    obs_spec_loglin, model_1_loglin = template_logwl_resample(obs_spec, model_1,
                                                               resampler=FluxConservingResampler())
    logger.info("...total time taken = %s seconds.", str(time.process_time() - tstart))
    print(f"...total time taken = {str(time.process_time() - tstart)} seconds.")

    logger.info("Resampling observed spectrum and second template to a common log-lambda wavelength"
                " scale...")
    print("Resampling observed spectrum and second template to a common log-lambda wavelength"
          " scale...")
    tstart = time.process_time()
    _, model_2_loglin = template_logwl_resample(obs_spec, model_2,
                                                 resampler=FluxConservingResampler())
    logger.info("...total time taken = %s seconds.", str(time.process_time() - tstart))
    print(f"...total time taken = {str(time.process_time() - tstart)} seconds.")

    # Each pixel represents how large a velocity step?
    ## TODO: I understand the first calculation (mostly, the need for the ln(10) factor I
    ## only partially understand)
    ## The other calculation I've seen is the second method, and it's (within 6 decimals)
    ## identical, but why? Would love to to understand the derivation of both of these better.
    delta_loglambda = math.log10(obs_spec_loglin.spectral_axis[1] /
                                  obs_spec_loglin.spectral_axis[0])
    # Convert speed of light into km/s so delta_vel is in km/s.
    # Should be c * ln(10) * delta_log(lambda)
    vel_per_pix = (speedoflight / 1000. * u.km / u.s *
                  delta_loglambda * math.log(10))
##    vel_per_pix2 = (speedoflight / 1000. * u.km / u.s *
##                   2. * (obs_spec_loglin.spectral_axis[4] - obs_spec_loglin.spectral_axis[3]) /
##                     (obs_spec_loglin.spectral_axis[4] + obs_spec_loglin.spectral_axis[3]))

    # Setup TODCOR structures that will contain the results: model_1 velocities,
    # model_2 velocities, and alpha (flux ratios.)
    # Length is 2n+1 to account for -shift-to-0, zero shift, and 0-to-+shift.
    # Each combination of model1 and model2 shift has a single corresponding alpha.
    # So the dimension is (2*npixshift+1, 2*npixshift+1)
    todcor_pix_dim = n_pix_shifts*2+1
    todcor_pixshifts = np.arange(todcor_pix_dim) - n_pix_shifts
    todcor_vals = np.zeros((todcor_pix_dim, todcor_pix_dim),
                                         dtype=np.float32)
    todcor_alphas = np.zeros((todcor_pix_dim, todcor_pix_dim),
                                         dtype=np.float32)

    logger.info("Calculating cross-correlations...")
    print("Calculating cross-correlations...")
    tstart = time.process_time()

    # Calculate cross-correlation function of observed spectrum versus model_1.
    corr1, lag1 = xcor1d(obs_spec_loglin.flux, model_1_loglin.flux)

    # Calculate cross-correlation function of observed spectrum versus model_2.
    corr2, lag2 = xcor1d(obs_spec_loglin.flux, model_2_loglin.flux)

    # Calculate cross-correlation function of model_1 versus model_2.
    corr12, lag12 = xcor1d(model_1_loglin.flux, model_2_loglin.flux)

    logger.info("...total time taken = %s seconds.", str(time.process_time() - tstart))
    print(f"...total time taken = {str(time.process_time() - tstart)} seconds.")

    # The output length of the cross-correlation must be at least twice as long as the
    # requested pixel shift, or else indexing into it to fill out the TODCOR result
    # array won't work.
    if len(corr1) != len(corr2) or len(corr1) != len(corr12):
        logger.error("The length of the cross-correlation result does not match"
                         " across observed spectrum and templates. Lengths of (obsxtemp1, "
                         "obsxtemp2, temp1xtemp2) "
                         "= (%d, %d, %d)", len(corr1), len(corr2), len(corr12))
        raise ValueError("The length of the cross-correlation result does not match"
                         " across observed spectrum and templates. Lengths of (obsxtemp1, "
                         "obsxtemp2, temp1xtemp2) "
                         f"= ({len(corr1)}, {len(corr2)}, {len(corr12)})")
    if len(corr1) < n_pix_shifts * 2:
        logger.error("The length of the cross-correlation result is not big enough"
                         " compared to the requested pixel shifts to test. Length of CCF"
                         " result = %d, asking for pixel shift +/- %d", len(corr1), n_pix_shifts)
        raise ValueError("The length of the cross-correlation result is not big enough"
                         " compared to the requested pixel shifts to test. Length of"
                         f" CCF result = {len(corr1)},"
                         f" asking for pixel shift +/- {n_pix_shifts}")

    # Trim the cross correlation arrays to only the number of pixel shifts of interest.
    where_keep = np.where((lag1 >= -1*n_pix_shifts) & (lag1 <= n_pix_shifts))[0]
    where_keep12 = np.where((lag1 >= -2*n_pix_shifts) & (lag1 <= 2*n_pix_shifts))[0]

    corr1 = corr1[where_keep]
    lag1 = lag1[where_keep]
    corr2 = corr2[where_keep]
    lag2 = lag2[where_keep]
    corr12 = corr12[where_keep12]
    lag12 = lag12[where_keep12]

    # Compute the TODCOR value following the equation A3 in Zucker & Mazeh 1994.
    logger.info("Calculating TODCOR values and populating the two-dimensional surface...")
    print("Calculating TODCOR values and populating the two-dimensional surface...")
    tstart = time.process_time()

    if fixed_alpha:
        # This is alpha_prime im the Appendix of Zucker & Mazeh 1994.
        alphap = np.std(model_2_loglin.flux) / np.std(model_1_loglin.flux) * fixed_alpha
        for i, corr1i in enumerate(corr1):
            for j, corr2j in enumerate(corr2):
                lag_i = lag1[i]
                lag_j = lag2[j]
                lag_ji = _find_rel_shift(lag_i, lag_j)
                lag_ji_ind = np.where(lag12 == lag_ji)[0]
                todcor_vals[i,j] = (corr1i + alphap * corr2j /
                                     ( (1. + 2.*alphap*corr12[lag_ji_ind] + alphap**2.)**0.5 ))

    logger.info("...total time taken = %s seconds.", str(time.process_time() - tstart))
    print(f"...total time taken = {str(time.process_time() - tstart)} seconds.")

    return (todcor_pixshifts, vel_per_pix, todcor_vals, todcor_alphas)
