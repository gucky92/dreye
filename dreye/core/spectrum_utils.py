"""
functions for creating particular spectra
or fitting particular spectra.
"""

import numpy as np
from scipy.stats import norm

from dreye.utilities import is_numeric
from dreye.core.spectrum import Spectrum
from dreye.core.spectral_measurement import SpectrumMeasurement


def fit_background(
    spectrum_measurement, background, return_fit=True, units=True,
    **kwargs
):
    """fit background spectrum to measurement of light sources.
    """

    assert isinstance(spectrum_measurement, SpectrumMeasurement)
    assert isinstance(background, Spectrum)

    return spectrum_measurement.fit(
        background, return_fit=return_fit, units=units, **kwargs)


def create_gaussian_spectrum(
    wavelengths, centers, std=10,
    intensity=1,
    units='microspectralphotonflux',
    cdf=None,
    background=None,
    filter=False,
    add_background=False,
    zero_cutoff=True,
):
    """
    get Gaussian spectra

    wavelengths : array-like
        Wavelength array in nm.
    centers : float, array-like
        The centers for each single wavelength Gaussian in nm.
    std : float, array-like
        The standard deviation for each single wavelength Gaussian in nm.
    intensity : float
        The intensity of the single wavelength Gaussians.
    """

    if background is None and (filter or add_background):
        raise ValueError(
            'must provide background if filter or add_background is True.'
        )
    elif isinstance(background, Spectrum):
        background = background.convert_to(units)(wavelengths)

    wavelengths = np.array(wavelengths)
    centers = np.array(centers)
    std = np.array(std)
    assert is_numeric(intensity)
    assert wavelengths.ndim == 1

    if background is not None:
        background = np.array(background)
        assert background.ndim == 1
        assert background.shape == wavelengths.shape

    if (centers.ndim == 1) or (std.ndim == 1):
        if centers.ndim == 1:
            centers = centers[None, :]
        if std.ndim == 1:
            std = std[None, :]

        wavelengths = wavelengths[:, None]
        if background is not None:
            background = background[:, None]

    if cdf is None:
        spectrum_array = norm.pdf(wavelengths, centers, std)
    elif cdf:
        spectrum_array = norm.cdf(wavelengths, centers, std)
    else:
        spectrum_array = 1 - norm.cdf(wavelengths, centers, std)

    if filter:
        # here spectrum array acts as a filter of the background illuminant
        spectrum_array = intensity * (
            background
            * spectrum_array
            / np.max(spectrum_array, axis=0, keepdims=True)
        )
    else:
        spectrum_array *= intensity / np.trapz(
            spectrum_array, np.squeeze(wavelengths), axis=0)[None]

    if add_background:
        spectrum_array += background

    if zero_cutoff:
        spectrum_array = np.clip(spectrum_array, 0, None)

    return Spectrum(
        spectrum_array,
        wavelengths,
        units=units,
        domain_axis=0
    )