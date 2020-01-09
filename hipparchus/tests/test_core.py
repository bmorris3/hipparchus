import pytest
import numpy as np
from scipy.ndimage import gaussian_filter1d

import astropy.units as u
from astropy.utils.data import download_file
from astropy.tests.helper import assert_quantity_allclose

from ..io import EchelleSpectrum, Template, Spectrum
from ..ccf import cross_corr

lkca4_url = "https://drive.google.com/uc?export=download&id=1x3nIg1P5tYFQqJrwEpQU11XQOs3ImH3v"
proxima_url = "https://drive.google.com/uc?export=download&id=1I7E1x1XRjcxXNQXiuaajb_Jz7Wn2N_Eo"
# T=3000 K, Molecule=TiO:
template_url = "https://drive.google.com/uc?export=download&id=1eGUBfk7Q9zaXgJQJtVFB6pit7cmoGCpn"


@pytest.mark.remote_data
@pytest.mark.parametrize("url,", [
    lkca4_url,
    proxima_url,
])
def test_ingest_e2ds(url):
    spectrum_path = download_file(url)
    spectrum = EchelleSpectrum.from_e2ds(spectrum_path)

    assert hasattr(spectrum.orders[0], 'wavelength')
    assert hasattr(spectrum.orders[0], 'flux')
    assert len(spectrum.orders) == 72
    assert spectrum.header['DATE-OBS'].startswith('2004')
    assert (spectrum.orders[1].wavelength.mean() >
            spectrum.orders[0].wavelength.mean())


@pytest.mark.remote_data
@pytest.mark.parametrize("url,", [
    lkca4_url,
    proxima_url,
])
def test_continuum_normalize(url):
    spectrum_path = download_file(url)
    spectrum = EchelleSpectrum.from_e2ds(spectrum_path)
    spectrum.continuum_normalize()

    # Confirm that each echelle order is roughly continuum normalized (roughly
    # distributed about unity):
    for order in spectrum.orders:
        assert abs(np.median(order.flux) - 1) < 1


@pytest.mark.remote_data
@pytest.mark.parametrize("url,", [
    proxima_url,
])
def test_end_to_end(url):
    spectrum_path = download_file(url)
    spectrum = EchelleSpectrum.from_e2ds(spectrum_path)

    spectrum.continuum_normalize()

    template_3000_tio_path = download_file(template_url)
    template_3000_tio = Template.from_npy(template_3000_tio_path)

    ccf = cross_corr(spectrum.nearest_order(6800), template_3000_tio)

    # Check that the TiO signal shows up at the radial velocity of the star
    assert_quantity_allclose(ccf.rv, -22 * u.km / u.s, atol=4 * u.km / u.s)

    # Check that the SNR of the detection is significant
    assert ccf.signal_to_noise > 5


@pytest.mark.parametrize("seed, snr,", [
    (42, 13),
    (1984, 10),
])
def test_ccf(seed, snr):
    # Make test reproducible:
    np.random.seed(seed)

    # Generate a wavelength grid:
    wl = np.arange(6800, 6900, 0.1)

    # Generate a pseudo-random absorption pattern
    absorption_pattern = gaussian_filter1d(0.1 * np.random.randn(len(wl)), 5)
    absorption_pattern = absorption_pattern - absorption_pattern.min()

    # Roll the absorption pattern offset by 3 indices
    index_offset = 3
    fl = np.ones_like(wl) - np.roll(absorption_pattern, index_offset)

    spectrum = Spectrum(wl, fl)

    emission = absorption_pattern / np.trapz(absorption_pattern, wl)
    template = Template(wl, emission)

    ccf = cross_corr(spectrum, template, start_lam=-10, end_lam=10,
                     n_steps=100)

    # Check that S/N is approximately correct:
    assert abs(ccf.signal_to_noise - snr) < 1

    # Check that the estimated radial velocity is equivalent to index offset
    ref_wl = 6800 * u.Angstrom
    measured_velocity_offset = (ccf.rv.to(u.Angstrom,
                                          u.doppler_optical(ref_wl)) -
                                ref_wl) / ((wl[1] - wl[0])*u.Angstrom)

    assert_quantity_allclose(measured_velocity_offset, index_offset, atol=1e-2)
