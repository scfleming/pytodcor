"""
.. module:: xcor1d
   :synopsis: Calculates cross-correlation of a spectrum with a model.
"""

import logging
import numpy as np
from scipy.signal import correlate, correlation_lags

logger = logging.getLogger("specxor")

def xcor1d(fluxes_1, fluxes_2):
    """
    Cross-correlates a one-dimensional set of values against a second set.

    :param fluxes_1: The first set of values to cross-correlate against.
    :type fluxes_1: astropy.units.quantity.Quantity
    :param fluxes_2: The second set of values to cross-correlate against.
    :type fluxes_2: astropy.units.quantity.Quantity
    :returns: tuple -- A tuple containing the array of linear cross-correlation values of
                       fluxes_1 with fluxes_2, and the array of lag indices.
    """

    # From Tonry & Davis, the 1-D cross-correlation is (noting DFT = "discrete Fourier transform"):
    # C = F(k) G(k)* / (N * sig_f * sig_g)
    # where:
    # F(k) = DFT(F); F = spectrum_1
    # G(k)* = complex conjugate of DFT(G); G = spectrum_2
    # N = number of overlapping bins
    # sig_f, sig_g = the RMS of each spectrum

    # Calculate the cross-correlation using the FFT method for speed.
    ccresult = correlate(fluxes_1, fluxes_2, mode="full", method="fft")
    # Divide by the rms of spectrum_1, rms of spectrum_2, and the number of bins.
    ## TODO: should be number of overlapping bins, may need to update to more than just len()
    denominator = np.std(fluxes_1) * np.std(fluxes_2) * len(fluxes_1)

    # Compute the lag indices.
    cclags = correlation_lags(len(fluxes_1), len(fluxes_2), mode="full")

    return ccresult/denominator, cclags
