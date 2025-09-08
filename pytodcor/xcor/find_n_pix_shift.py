from astropy.constants import c
import numpy as np
def find_n_pix_shift(obs_spec, max_vel, overshoot_factor):
    speed_of_light = c.to("km/s").value
    min_wavelength, max_wavelength = obs_spec.wavelength[0], obs_spec.wavelength[-1]

    delta_lambda_min = min_wavelength * (max_vel / speed_of_light)
    delta_lambda_max = max_wavelength * (max_vel / speed_of_light)
    delta_lambda = max_vel / speed_of_light * max_wavelength
    wavelength_diffs = np.diff(obs_spec.wavelength)

    n_pix_shifts = np.ceil(delta_lambda / min(wavelength_diffs)) * overshoot_factor
    return int(n_pix_shifts.value)