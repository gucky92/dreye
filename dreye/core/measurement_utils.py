"""
utility functions for spectrum measurements
"""

import numpy as np

# dreye modules
from dreye.utilities import has_units, is_numeric, asarray
from dreye.constants import ureg
from dreye.core.domain import Domain
from dreye.core.signal import Signal
from dreye.core.spectrum import AbstractSpectrum, Spectrum
from dreye.core.spectral_measurement import CalibrationSpectrum, \
    MeasuredSpectrum


def create_calibration_spectrum(
    spectrum_array, wavelengths,
    area, area_units='cm ** 2',
    **kwargs
):
    """convenience function for creating calibration spectrum.
    """

    return CalibrationSpectrum(
        spectrum_array, wavelengths, area=area,
        area_units=area_units,
        **kwargs
    )


def convert_measurement(
    signal, calibration, integration_time,
    units='microspectralphotonflux',
    spectrum_cls=Spectrum, **kwargs
):
    """
    function to convert photon count signal into spectrum.
    """

    assert isinstance(signal, Signal)
    assert isinstance(calibration, CalibrationSpectrum)

    area = calibration.area

    if not has_units(integration_time):
        integration_time = integration_time * ureg('s')

    if not is_numeric(integration_time):
        assert signal.ndim == 2
        integration_time = np.expand_dims(
            asarray(integration_time), signal.domain_axis
        ) * integration_time.units

    # units are tracked
    spectrum = (signal * calibration)
    spectrum /= (integration_time * area)
    spectrum = spectrum.piecewise_gradient

    return spectrum_cls(spectrum, units=units, **kwargs)


def create_measured_spectrum(
    spectrum_array, inputs, wavelengths,
    calibration, integration_time,
    axis=0,
    units='microspectralphotonflux',
    input_units='V',
    is_mole=False, **kwargs
):
    """

    Parameters
    ----------
    spectrum_array : array-like
        array of photon counts across wavelengths for each input
    inputs : array-like
        array of inputs
    wavelengths : array-like
        array of wavelengths in nanometers.
    calibration : CalibrationSpectrum
        Calibration spectrum
    integration_times : array-like
        integration times in seconds.
    axis : int
        axis of wavelengths in spectrum_array
    units : str
        units to convert to
    input_units : str
        units of inputs.
    """

    # create labels
    labels = Domain(
        inputs, units=input_units
    )

    spectrum = AbstractSpectrum(
        spectrum_array,
        wavelengths,
        domain_axis=axis,
    )
    if is_mole:
        spectrum = spectrum * ureg('mole')

    return convert_measurement(
        spectrum,
        calibration=calibration,
        integration_time=integration_time,
        labels=labels,
        units=units,
        spectrum_cls=MeasuredSpectrum, **kwargs
    )


def create_measured_spectra(
    *args, units='uE',
    **kwargs
):
    """convenience function
    """

    return create_measured_spectrum(*args, **kwargs).to_measured_spectra(
        units=units
    )
