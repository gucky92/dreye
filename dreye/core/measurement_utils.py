"""
utility functions for spectrum measurements
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from dreye.utilities import has_units, is_numeric, asarray
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.core.domain import Domain
from dreye.core.signal import _SignalMixin, _Signal2DMixin
from dreye.core.spectrum import IntensitySpectra, DomainSpectrum
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum,
    MeasuredSpectraContainer
)


def convert_measurement(
    signal, calibration=None, integration_time=None,
    area=None,
    units='uE',
    spectrum_cls=IntensitySpectra,
    **kwargs
):
    """
    function to convert photon count signal into spectrum.
    """

    assert isinstance(signal, _SignalMixin)

    if calibration is None:
        calibration = CalibrationSpectrum(
            np.ones(signal.domain.size),
            signal.domain,
            area=area
        )

    if area is None:
        assert isinstance(calibration, CalibrationSpectrum)
        area = calibration.area
    else:
        CalibrationSpectrum(
            calibration,
            domain=signal.domain,
            area=area
        )
        area = calibration.area

    if integration_time is None:
        integration_time = ureg('s')  # assumes 1 seconds integration time

    if not has_units(integration_time):
        integration_time = integration_time * ureg('s')

    if not is_numeric(integration_time):
        integration_time = np.expand_dims(
            integration_time.magnitude, signal.domain_axis
        ) * integration_time.units

    # units are tracked
    spectrum = (signal * calibration)
    spectrum = spectrum / (integration_time * area)
    spectrum = spectrum.piecewise_gradient

    return spectrum_cls(spectrum, units=units, **kwargs)


def create_measured_spectrum(
    spectrum_array, output,
    wavelengths,
    calibration=None,
    integration_time=None,
    area=None,
    units='uE',
    output_units='V',
    is_mole=False,
    zero_intensity_bound=None,
    max_intensity_bound=None,
    assume_contains_output_bounds=True,
    resolution=None
):
    """
    Parameters
    ----------
    spectrum_array : array-like
        array of photon counts across wavelengths for each output
        (wavelength x output labels).
    output : array-like
        array of output in ascending order.
    wavelengths : array-like
        array of wavelengths in nanometers in ascending order.
    calibration : CalibrationSpectrum or array-like
        Calibration spectrum
    integration_times : array-like
        integration times in seconds.
    axis : int
        axis of wavelengths in spectrum_array
    units : str
        units to convert to
    output_units : str
        units of output.
    """
    # create labels
    spectrum = DomainSpectrum(
        spectrum_array,
        domain=wavelengths,
        labels=Domain(output, units=output_units)
    )
    if assume_contains_output_bounds:
        intensities = spectrum.magnitude.sum(0)
        if intensities[0] > intensities[-1]:
            if zero_intensity_bound is not None:
                zero_intensity_bound = spectrum.labels.end
            if max_intensity_bound is not None:
                max_intensity_bound = spectrum.labels.start
        else:
            if zero_intensity_bound is not None:
                zero_intensity_bound = spectrum.labels.start
            if max_intensity_bound is not None:
                max_intensity_bound = spectrum.labels.end
    if is_mole:
        spectrum = spectrum * ureg('mol')

    return convert_measurement(
        spectrum,
        calibration=calibration,
        integration_time=integration_time,
        units=units,
        area=area,
        spectrum_cls=MeasuredSpectrum,
        zero_intensity_bound=zero_intensity_bound,
        max_intensity_bound=max_intensity_bound,
        resolution=resolution
    )


def create_measured_spectra(
    spectrum_arrays,
    output_arrays,
    wavelengths,
    calibration,
    integration_time,
    area=None,
    units='uE',
    output_units='V',
    is_mole=False,
    assume_contains_output_bounds=True,
    resolution=None
):
    """convenience function
    """

    measured_spectra = []
    for spectrum_array, output in zip(spectrum_arrays, output_arrays):
        measured_spectrum = create_measured_spectrum(
            spectrum_array, output, wavelengths,
            calibration=calibration,
            integration_time=integration_time, area=area,
            units=units, output_units=output_units,
            is_mole=is_mole,
            resolution=resolution,
            assume_contains_output_bounds=assume_contains_output_bounds
        )
        measured_spectra.append(measured_spectrum)

    return MeasuredSpectraContainer(measured_spectra)


def get_led_spectra_container(
    led_spectra=None,  # wavelengths x LED (ignores units)
    intensity_bounds=(0, 100),  # two-tuple of min and max intensity
    wavelengths=None,  # wavelengths (two-tuple or array-like)
    output_bounds=None,  # two-tuple of min and max output
    resolution=None,  # array-like
    intensity_units='uE',  # units
    output_units=None,
    transform_func=None  # callable
):
    """
    Convenience function to created measured spectra container from
    LED spectra and intensity bounds.
    """
    # create fake LEDs
    if led_spectra is None or is_numeric(led_spectra):
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.1)
        if led_spectra is None:
            centers = np.arange(350, 700, 50)  # 7 LEDs
        else:
            centers = np.arange(350, 650, int(led_spectra))
        led_spectra = norm.pdf(wavelengths[:, None], centers, 20)
    # check if we can obtain wavelengths
    if wavelengths is None:
        if hasattr(led_spectra, 'domain'):
            wavelengths = led_spectra.domain
        elif hasattr(led_spectra, 'wavelengths'):
            wavelengths = led_spectra.wavelengths
        elif isinstance(led_spectra, (pd.DataFrame, pd.Index)):
            wavelengths = asarray(led_spectra.index)
        else:
            raise DreyeError("Must provide wavelengths.")
    if isinstance(led_spectra, _Signal2DMixin):
        led_spectra = led_spectra(wavelengths)
        led_spectra.domain_axis = 0

    led_spectra = asarray(led_spectra)
    led_spectra /= np.trapz(led_spectra, wavelengths, axis=0)

    measured_spectra = []
    for idx, led_spectrum in enumerate(led_spectra.T):
        # always do 100 hundred steps
        led_spectrum = led_spectrum * np.linspace(*intensity_bounds, 100)[None]
        if output_bounds is None:
            output = np.linspace(*intensity_bounds, 100)
        elif transform_func is not None:
            output = transform_func(np.linspace(*intensity_bounds, 100))
        else:
            output = np.linspace(*output_bounds, 100)

        if intensity_units in MeasuredSpectrum._unit_mappings:
            units = intensity_units
        elif isinstance(intensity_units, str):
            units = ureg(intensity_units).units / ureg('nm').units
        elif has_units(intensity_units):
            units = intensity_units.units / ureg('nm').units
        elif units is None:
            units = 'uE'  # assumes in microspectralphotonflux
        else:
            # assumes is ureg.Unit
            units = intensity_units / ureg('nm').units

        measured_spectrum = MeasuredSpectrum(
            values=led_spectrum,
            domain=wavelengths,
            labels=output,
            labels_units=output_units,
            units=units,
            resolution=resolution
        )
        measured_spectra.append(measured_spectrum)

    return MeasuredSpectraContainer(measured_spectra)
