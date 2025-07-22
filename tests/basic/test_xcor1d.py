import pytest
import numpy as np
from scipy.stats import norm
from pytodcor import xcor1d

def test_xcor1d_identical_gauss_no_shift():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 50., 2.)
    f2_x = np.arange(100)
    f2_y = norm.pdf(f1_x, 50., 2.)

    # Calculate the one-dimensional cross-correlation.
    corr, lag = xcor1d(f1_y, f2_y)

    # Find the peak value (without fitting/interpolating between points):
    peak_ind = np.argmax(corr)
    peak_val = corr[peak_ind]
    peak_lag = lag[peak_ind]

    assert peak_lag == 0
