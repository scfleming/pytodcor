import pytest
import numpy as np
from astropy import units as u
from scipy.stats import norm
from scipy.signal import find_peaks
from specutils.spectra import Spectrum
from pytodcor.xcor.todcor import todcor
from pytodcor.lib.spectrum import PytodcorSpectrum

class Test01:
    def setup_class(cls):
        # Generate the Gaussians to use.
        cls.gauss_1_peak = 5380.
        cls.gauss_2_peak = 5420.
        cls.xvals = np.linspace(5300., 5500., 1000)
        cls.g1_y = norm.pdf(cls.xvals, cls.gauss_1_peak, 4.)
        cls.g2_y = norm.pdf(cls.xvals, cls.gauss_2_peak, 4.)
        cls.obs_y = cls.g1_y + cls.g2_y
        cls.mod1_y = cls.g1_y
        cls.mod2_y = cls.g2_y

        cls.n_pix_apart = len(np.where((cls.xvals >= cls.gauss_1_peak) & (cls.xvals <= cls.gauss_2_peak))[0])

        cls.obs_spec1d = Spectrum(flux=cls.obs_y * u.dimensionless_unscaled,
                                  spectral_axis=cls.xvals * u.angstrom)
        cls.obs_spec = PytodcorSpectrum(name="Case_01_Obs", air_or_vac="vacuum")
        cls.obs_spec.add_spec_part(cls.obs_spec1d)

        cls.mod1_spec1d = Spectrum(flux=cls.mod1_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod1_spec = PytodcorSpectrum(name="Case_01_Mod1", air_or_vac="vacuum")
        cls.mod1_spec.add_spec_part(cls.mod1_spec1d)

        cls.mod2_spec1d = Spectrum(flux=cls.mod2_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod2_spec = PytodcorSpectrum(name="Case_01_Mod2", air_or_vac="vacuum")
        cls.mod2_spec.add_spec_part(cls.mod2_spec1d)

        cls.todcor_pixshifts, cls.vel_per_pix, cls.todcor_vals, cls.todcor_alphas = todcor(
            cls.obs_spec.parts[0], cls.mod1_spec.parts[0], cls.mod2_spec.parts[0], 400,
            fixed_alpha=1., vel_range=[-500., 500.]
        )
        cls.x_fix_shift_1 = 0
        cls.x_fix_shift_2 = cls.n_pix_apart
        cls.y_fix_shift_1 = 0
        cls.y_fix_shift_2 = -1 * cls.n_pix_apart
        cls.where_x1 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_1)[0][0]
        cls.where_x2 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_2)[0][0]
        cls.where_y1 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_1)[0][0]
        cls.where_y2 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_2)[0][0]
    def test_primary_zero_shift(self):
        peak1_pri_shift0, peak2_pri_shift0 = find_peaks(self.todcor_vals[self.where_x1, :])[0] - 400
        assert peak1_pri_shift0 == pytest.approx(-203, abs=1)
        assert peak2_pri_shift0 == pytest.approx(0, abs=1)
    def test_primary_nonzero_shift(self):
        peak1_pri_shift200, peak2_pri_shift200 = find_peaks(self.todcor_vals[self.where_x2, :])[0] - 400
        assert peak1_pri_shift200 == pytest.approx(-203, abs=1)
        assert peak2_pri_shift200 == pytest.approx(0, abs=1)
    def test_secondary_zero_shift(self):
        peak1_sec_shift0, peak2_sec_shift0 = find_peaks(self.todcor_vals[:, self.where_y1])[0] - 400
        assert peak1_sec_shift0 == pytest.approx(0, abs=1)
        assert peak2_sec_shift0 == pytest.approx(203, abs=1)
    def test_secondary_nonzero_shift(self):
        peak1_sec_shift200, peak2_sec_shift200 = find_peaks(self.todcor_vals[:, self.where_y2])[0] - 400
        assert peak1_sec_shift200 == pytest.approx(0, abs=1)
        assert peak2_sec_shift200 == pytest.approx(203, abs=1)



class Test02:
    def setup_class(cls):
        # Generate the Gaussians to use.
        cls.gauss_1_peak = 5370.
        cls.gauss_2_peak = 5430.
        cls.xvals = np.linspace(5300., 5500., 1000)
        cls.g1_y = norm.pdf(cls.xvals, cls.gauss_1_peak, 4.)
        cls.g2_y = norm.pdf(cls.xvals, cls.gauss_2_peak, 4.)
        cls.obs_y = cls.g1_y + cls.g2_y
        cls.mod1_y = cls.g1_y
        cls.mod2_y = cls.g2_y

        cls.n_pix_apart = len(np.where((cls.xvals >= cls.gauss_1_peak) & (cls.xvals <= cls.gauss_2_peak))[0])

        cls.obs_spec1d = Spectrum(flux=cls.obs_y * u.dimensionless_unscaled,
                                  spectral_axis=cls.xvals * u.angstrom)
        cls.obs_spec = PytodcorSpectrum(name="Case_02_Obs", air_or_vac="vacuum")
        cls.obs_spec.add_spec_part(cls.obs_spec1d)

        cls.mod1_spec1d = Spectrum(flux=cls.mod1_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod1_spec = PytodcorSpectrum(name="Case_02_Mod1", air_or_vac="vacuum")
        cls.mod1_spec.add_spec_part(cls.mod1_spec1d)

        cls.mod2_spec1d = Spectrum(flux=cls.mod2_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod2_spec = PytodcorSpectrum(name="Case_02_Mod2", air_or_vac="vacuum")
        cls.mod2_spec.add_spec_part(cls.mod2_spec1d)

        cls.todcor_pixshifts, cls.vel_per_pix, cls.todcor_vals, cls.todcor_alphas = todcor(
            cls.obs_spec.parts[0], cls.mod1_spec.parts[0], cls.mod2_spec.parts[0], 400,
            fixed_alpha=1., vel_range=[-500., 500.]
        )
        cls.x_fix_shift_1 = 0
        cls.x_fix_shift_2 = cls.n_pix_apart
        cls.y_fix_shift_1 = 0
        cls.y_fix_shift_2 = -1 * cls.n_pix_apart
        cls.where_x1 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_1)[0][0]
        cls.where_x2 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_2)[0][0]
        cls.where_y1 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_1)[0][0]
        cls.where_y2 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_2)[0][0]

    def test_primary_zero_shift(self):
        peak1_pri_zero_shift, peak2_pri_zero_shift = find_peaks(self.todcor_vals[self.where_x1, :])[0] - 400
        assert peak1_pri_zero_shift == pytest.approx(-304, abs=1)
        assert peak2_pri_zero_shift == pytest.approx(0, abs=2)
    def test_primary_nonzero_shift(self):
        peak1_pri_nonzero_shift, peak2_pri_nonzero_shift = find_peaks(self.todcor_vals[self.where_x2, :])[0] - 400
        assert peak1_pri_nonzero_shift == pytest.approx(-304, abs=1)
        assert peak2_pri_nonzero_shift == pytest.approx(0, abs=2)
    def test_secondary_zero_shift(self):
        peak1_sec_zero_shift, peak2_sec_zero_shift = find_peaks(self.todcor_vals[:, self.where_y1])[0] - 400
        assert peak1_sec_zero_shift == pytest.approx(0, abs=2)
        assert peak2_sec_zero_shift == pytest.approx(304, abs=1)
    def test_secondary_nonzero_shift(self):
        peak1_sec_nonzero_shift, peak2_sec_nonzero_shift = find_peaks(self.todcor_vals[:, self.where_y2])[0] - 400
        assert peak1_sec_nonzero_shift == pytest.approx(0, abs=2)
        assert peak2_sec_nonzero_shift == pytest.approx(304, abs=1)

class Test03:
    def setup_class(cls):
        # Generate the Gaussians to use (identical peaks).
        cls.gauss_1_peak = 5400.
        cls.gauss_2_peak = 5400.
        cls.xvals = np.linspace(5300., 5500., 1000)
        cls.g1_y = norm.pdf(cls.xvals, cls.gauss_1_peak, 4.)
        cls.g2_y = norm.pdf(cls.xvals, cls.gauss_2_peak, 4.)
        cls.obs_y = cls.g1_y + cls.g2_y
        cls.mod1_y = cls.g1_y
        cls.mod2_y = cls.g2_y

        cls.n_pix_apart = len(np.where((cls.xvals >= cls.gauss_1_peak) & (cls.xvals <= cls.gauss_2_peak))[0])

        cls.obs_spec1d = Spectrum(flux=cls.obs_y * u.dimensionless_unscaled,
                                  spectral_axis=cls.xvals * u.angstrom)
        cls.obs_spec = PytodcorSpectrum(name="Case_03_Obs", air_or_vac="vacuum")
        cls.obs_spec.add_spec_part(cls.obs_spec1d)

        cls.mod1_spec1d = Spectrum(flux=cls.mod1_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod1_spec = PytodcorSpectrum(name="Case_03_Mod1", air_or_vac="vacuum")
        cls.mod1_spec.add_spec_part(cls.mod1_spec1d)

        cls.mod2_spec1d = Spectrum(flux=cls.mod2_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod2_spec = PytodcorSpectrum(name="Case_03_Mod2", air_or_vac="vacuum")
        cls.mod2_spec.add_spec_part(cls.mod2_spec1d)

        cls.todcor_pixshifts, cls.vel_per_pix, cls.todcor_vals, cls.todcor_alphas = todcor(
            cls.obs_spec.parts[0], cls.mod1_spec.parts[0], cls.mod2_spec.parts[0], 400,
            fixed_alpha=1., vel_range=[-500., 500.]
        )
        cls.x_fix_shift_1 = 0
        cls.x_fix_shift_2 = cls.n_pix_apart
        cls.y_fix_shift_1 = 0
        cls.y_fix_shift_2 = -1 * cls.n_pix_apart
        cls.where_x1 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_1)[0][0]
        cls.where_x2 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_2)[0][0]
        cls.where_y1 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_1)[0][0]
        cls.where_y2 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_2)[0][0]
    # NO TOLERANCE -- PEAK SHOULD BE ZERO FOR ALL
    def test_primary_zero_shift(self):
        peak = (find_peaks(self.todcor_vals[self.where_x1, :])[0] - 400)[0]
        assert peak == 0

    def test_primary_nonzero_shift(self):
        peak = (find_peaks(self.todcor_vals[self.where_x2, :])[0] - 400)[0]
        assert peak == 0

    def test_secondary_zero_shift(self):
        peak = (find_peaks(self.todcor_vals[:, self.where_y1])[0] - 400)[0]
        assert peak == 0

    def test_secondary_nonzero_shift(self):
        peak = (find_peaks(self.todcor_vals[:, self.where_y2])[0] - 400)[0]
        assert peak == 0
class Test04:
    def setup_class(cls):
        # Generate the Gaussians to use (smaller width, unequal amplitudes, different spectral range)
        cls.gauss_1_peak = 4080.
        cls.gauss_2_peak = 4120.
        cls.xvals = np.linspace(4000., 4200., 1000)
        cls.g1_y = norm.pdf(cls.xvals, cls.gauss_1_peak, 1.)
        cls.g2_y = 0.1 * norm.pdf(cls.xvals, cls.gauss_2_peak, 1.)  # Second Gaussian has 10% amplitude
        cls.obs_y = cls.g1_y + cls.g2_y
        cls.mod1_y = cls.g1_y
        cls.mod2_y = cls.g2_y

        cls.n_pix_apart = len(np.where((cls.xvals >= cls.gauss_1_peak) & (cls.xvals <= cls.gauss_2_peak))[0])

        cls.obs_spec1d = Spectrum(flux=cls.obs_y * u.dimensionless_unscaled,
                                  spectral_axis=cls.xvals * u.angstrom)
        cls.obs_spec = PytodcorSpectrum(name="Case_04_Obs", air_or_vac="vacuum")
        cls.obs_spec.add_spec_part(cls.obs_spec1d)

        cls.mod1_spec1d = Spectrum(flux=cls.mod1_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod1_spec = PytodcorSpectrum(name="Case_04_Mod1", air_or_vac="vacuum")
        cls.mod1_spec.add_spec_part(cls.mod1_spec1d)

        cls.mod2_spec1d = Spectrum(flux=cls.mod2_y * u.dimensionless_unscaled,
                                   spectral_axis=cls.xvals * u.angstrom)
        cls.mod2_spec = PytodcorSpectrum(name="Case_04_Mod2", air_or_vac="vacuum")
        cls.mod2_spec.add_spec_part(cls.mod2_spec1d)

        cls.todcor_pixshifts, cls.vel_per_pix, cls.todcor_vals, cls.todcor_alphas = todcor(
            cls.obs_spec.parts[0], cls.mod1_spec.parts[0], cls.mod2_spec.parts[0], 400,
            fixed_alpha=1., vel_range=[-500., 500.]
        )
        cls.x_fix_shift_1 = 0
        cls.x_fix_shift_2 = cls.n_pix_apart
        cls.y_fix_shift_1 = 0
        cls.y_fix_shift_2 = -1 * cls.n_pix_apart
        cls.where_x1 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_1)[0][0]
        cls.where_x2 = np.where(cls.todcor_pixshifts == cls.x_fix_shift_2)[0][0]
        cls.where_y1 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_1)[0][0]
        cls.where_y2 = np.where(cls.todcor_pixshifts == cls.y_fix_shift_2)[0][0]

    def test_primary_zero_shift(self):
        peak1_pri_zero_shift, peak2_pri_zero_shift = find_peaks(self.todcor_vals[self.where_x1, :])[0] - 400
        assert peak1_pri_zero_shift == pytest.approx(-204, abs=1)
        assert peak2_pri_zero_shift == pytest.approx(0, abs=1)

    def test_primary_nonzero_shift(self):
        peak1_pri_nonzero_shift, peak2_pri_nonzero_shift = find_peaks(self.todcor_vals[self.where_x2, :])[0] - 400
        assert peak1_pri_nonzero_shift == pytest.approx(-204, abs=1)
        assert peak2_pri_nonzero_shift == pytest.approx(0, abs=1)

    def test_secondary_zero_shift(self):
        peak1_sec_zero_shift, peak2_sec_zero_shift = find_peaks(self.todcor_vals[:, self.where_y1])[0] - 400
        assert peak1_sec_zero_shift == pytest.approx(0, abs=1)
        assert peak2_sec_zero_shift == pytest.approx(204, abs=1)

    def test_secondary_nonzero_shift(self):
        peak1_sec_nonzero_shift, peak2_sec_nonzero_shift = find_peaks(self.todcor_vals[:, self.where_y2])[0] - 400
        assert peak1_sec_nonzero_shift == pytest.approx(0, abs=1)
        assert peak2_sec_nonzero_shift == pytest.approx(204, abs=1)