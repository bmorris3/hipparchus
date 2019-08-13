import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy.stats import mad_std

__all__ = ['CCF', 'cross_corr']


def cross_corr(spectrum, template, start_lam=-2, end_lam=2, n_steps=1000,
               sigma=None, spread_factor=10):
    """
    Cross-correlation of the spectrum and template.

    Parameters
    ----------
    spectrum : `~hipparchus.Spectrum`
        Spectrum object (may be an order of an echelle spectrum).
    template : `~hipparchus.Template`
        Template object to correlate against the spectrum.

    Returns
    -------
    ccf : `~hipparchus.CCF`
        Cross-correlation object.
    """

    dx_range = np.linspace(start_lam, end_lam, n_steps)[:, np.newaxis]

    shifted_wavelengths = spectrum.wavelength - dx_range
    lam0 = spectrum.wavelength.mean() * u.Angstrom
    velocities = (lam0 + dx_range*u.Angstrom).to(u.km/u.s,
                                                 u.doppler_optical(lam0))

    if sigma is None:
        sigma = ((spectrum.wavelength[1] - spectrum.wavelength[0]) /
                 (template.wavelength[1] - template.wavelength[0]) /
                 spread_factor)

    smoothed_emission = gaussian_filter1d(template.emission, sigma)
    T = interp1d(template.wavelength, smoothed_emission, bounds_error=False,
                 fill_value=0)(shifted_wavelengths)

    x = np.repeat(spectrum.flux[:, np.newaxis], n_steps, axis=1).T

    ccf = np.average(x, weights=T, axis=1)
    return CCF(velocities, ccf)


class CCF(object):
    """
    Storage object for cross-correlation functions
    """
    def __init__(self, velocities, ccf):
        self.velocities = velocities
        self.ccf = ccf

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.velocities, self.ccf/np.median(self.ccf))
        ax.set_xlabel('$\Delta v$ [km/s]')  # noqa
        ax.set_ylabel('CCF')
        return ax

    @property
    def rv(self):
        return self.velocities[np.argmin(self.ccf)]

    @property
    def signal_to_noise(self):
        return (np.median(self.ccf) - self.ccf.min()) / mad_std(self.ccf)
