import pytest
import numpy as np
from scipy.stats import norm
from pytodcor.xcor.xcor1d import xcor1d

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
def test_xcor1d_half_gauss_no_shift():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 50., 2.)
    f2_x = np.arange(100)
    f2_y = norm.pdf(f1_x, 50., 2.)
    f2_y *= 0.5

    # Calculate the one-dimensional cross-correlation.
    corr, lag = xcor1d(f1_y, f2_y)

    # Find the peak value (without fitting/interpolating between points):
    peak_ind = np.argmax(corr)
    peak_val = corr[peak_ind]
    peak_lag = lag[peak_ind]

    assert peak_lag == 0
def test_xcor1d_identical_gauss_shifted():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 30., 2.)
    f2_x = np.arange(100)
    f2_y = norm.pdf(f1_x, 60., 2.)

    # Calculate the one-dimensional cross-correlation.
    corr, lag = xcor1d(f1_y, f2_y)

    # Find the peak value (without fitting/interpolating between points):
    peak_ind = np.argmax(corr)
    peak_val = corr[peak_ind]
    peak_lag = lag[peak_ind]

    assert peak_lag == -30
def test_xcor1d_identical_double_gauss_shifted():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 30., 2.)
    f2_x = np.arange(100)
    f2_y = norm.pdf(f1_x, 60., 2.)
    f1_y += f2_y

    # Calculate the one-dimensional cross-correlation.
    corr, lag = xcor1d(f1_y, f2_y)

    # Find the peak value (without fitting/interpolating between points):
    corr_sort_ind = np.flip(np.argsort(corr))
    peak_val, peak_val_2 = corr[corr_sort_ind[0:2]]
    peak_lag, peak_lag_2 = lag[corr_sort_ind[0:2]]
    assert -30 in [peak_lag, peak_lag_2] and 0 in [peak_lag, peak_lag_2]
def test_xcor1d_gauss_edge_shift():
    # Generate the Gaussians to use, each at left or right edge.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 5., 2.)
    f2_y = norm.pdf(f1_x, 95., 2.)
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == -90
def test_xcor1d_rectangular_shifted():
    # Generate the Gaussians to use.
    f1_y = np.zeros(100)
    f1_y[40:50] = 0.1
    f2_y = np.zeros(100)
    f2_y[60:70] = 0.1
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == -20
def test_xcor1d_one_rectangular_shifted():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 60., 2.)
    f2_y = np.zeros(100)
    f2_y[10:20] = 0.1
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == pytest.approx(45,abs = 5) #since it is rectangular, the peak should be in this range
def test_xcor1d_identical_gauss_negative_shift():
    # Generate the Gaussians to use.
    f1_x = np.arange(100)
    f1_y = norm.pdf(f1_x, 80., 2.)
    f2_x = np.arange(100)
    f2_y = norm.pdf(f1_x, 20., 2.)
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == 60
def test_xcor1d_identical_small():
    f1_y = norm.pdf(np.arange(5), 2, 1.)
    f2_y = norm.pdf(np.arange(5), 2, 1.)
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == 0
def test_xcor1d_identical_small_shifted():
    f1_y = norm.pdf(np.arange(5), 4, 1.)
    f2_y = norm.pdf(np.arange(5), 1, 1.)
    corr, lag = xcor1d(f1_y, f2_y)
    peak_ind = np.argmax(corr)
    peak_lag = lag[peak_ind]
    assert peak_lag == 3