import pytest
import numpy as np
from astropy import units as u
from scipy.stats import norm
from skimage.feature import peak_local_max
from specutils.spectra import Spectrum
from pytodcor.xcor.todcor import todcor
from pytodcor.lib.spectrum import PytodcorSpectrum

class TestCase01:
    """
    Identical x-axis, Identical Gaussians, No Template Shifts
    """
    def setup_class(self):
        # Generate the Gaussians to use.
        gauss_1_peak = 5380.
        gauss_2_peak = 5420.
        xvals = np.linspace(5300., 5500., 1000)
        g1_y = norm.pdf(xvals, gauss_1_peak, 4.)
        g2_y = norm.pdf(xvals, gauss_2_peak, 4.)
        # Double-peaked Gaussian
        obs_y = g1_y + g2_y
        # Two single-peaked Gaussians
        mod1_y = g1_y
        mod2_y = g2_y

        # This is the separation before log-linear resampling.
        self.n_pix_apart = len(np.where((xvals >= gauss_1_peak) & (xvals <= gauss_2_peak))[0])
        # This is the expected shift after log-linear resampling.
        self.n_pix_apart_loglin = 203

        obs_spec1d = Spectrum(flux=obs_y * u.dimensionless_unscaled,
                                  spectral_axis=xvals * u.angstrom)
        obs_spec = PytodcorSpectrum(name="Case_01_Obs", air_or_vac="vacuum")
        obs_spec.add_spec_part(obs_spec1d)

        mod1_spec1d = Spectrum(flux=mod1_y * u.dimensionless_unscaled,
                                   spectral_axis=xvals * u.angstrom)
        mod1_spec = PytodcorSpectrum(name="Case_01_Mod1", air_or_vac="vacuum")
        mod1_spec.add_spec_part(mod1_spec1d)

        mod2_spec1d = Spectrum(flux=mod2_y * u.dimensionless_unscaled,
                                   spectral_axis=xvals * u.angstrom)
        mod2_spec = PytodcorSpectrum(name="Case_01_Mod2", air_or_vac="vacuum")
        mod2_spec.add_spec_part(mod2_spec1d)

        self.todcor_pixshifts, self.vel_per_pix, self.todcor_vals, self.todcor_alphas = todcor(
            obs_spec.parts[0], mod1_spec.parts[0], mod2_spec.parts[0], 400,
            fixed_alpha=1., vel_range=[-500., 500.]
        )

    def test_case_01(self):
        # Find local max peaks within the two-dimensional TODCOR array.
        # The return are indices, which can be indexed into the "pixshifts" to translate into lag values.
        twod_peaks = peak_local_max(self.todcor_vals, min_distance=4)
        # Strongest peak at (0,0)
        assert np.array_equal(self.todcor_pixshifts[twod_peaks[0]], np.asarray([0, 0]))
        # Second strongest peak at (n_pix_apart_loglin, -1*n_pix_apart_loglin)
        assert np.array_equal(self.todcor_pixshifts[twod_peaks[1]],
                                  np.asarray([self.n_pix_apart_loglin, -1*self.n_pix_apart_loglin]))
        # Third and fourth strongest peaks at (0, -1* n_pix_apart_loglin) or (n_pix_apart_loglin, 0), don't assume an order.
        assert (
            np.array_equal(self.todcor_pixshifts[twod_peaks[2]], np.asarray([0, -1*self.n_pix_apart_loglin]))
            or
            np.array_equal(self.todcor_pixshifts[twod_peaks[2]], np.asarray([self.n_pix_apart_loglin, 0]))
            )

        assert (
            np.array_equal(self.todcor_pixshifts[twod_peaks[3]], np.asarray([0, -1*self.n_pix_apart_loglin]))
            or
            np.array_equal(self.todcor_pixshifts[twod_peaks[3]], np.asarray([self.n_pix_apart_loglin, 0]))
            )
