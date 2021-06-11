"""Class to define photoreceptor/capture model
"""

from dreye.utilities.common import is_listlike
import warnings
import copy
from abc import ABC

from scipy.stats import norm
import numpy as np

from dreye.core.signal import Signals, Signal
from dreye.utilities.abstract import inherit_docstrings
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.utilities import (
    is_callable, has_units, is_numeric, 
    is_signallike, is_integer
)
from dreye.core.opsin_template import govardovskii2000_template


RELATIVE_SENSITIVITY_SIGNIFICANT = 1e-1
WL_PEAK_SPACE = 150


def create_photoreceptor_model(
    sensitivity=None, wavelengths=None, filterfunc=None, labels=None,
    photoreceptor_type='linear', **kwargs
):
    """
    Create an arbitrary photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    photoreceptor_type : str ({'linear', 'log', 'hyperbolic'}), optional
        The photoreceptor model.
    kwargs : dict, optional
        A dictionary that is directly passed to the photoreceptor class
        instantiation.

    Returns
    -------
    object : `Photoreceptor` class
        A photoreceptor instance which has the `capture` and `excitation`
        method implemented.

    See Also
    --------
    dreye.Photoreceptor
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    This is a convenience function to create arbitrary photoreceptor models.
    It is also possible to create a photoreceptor model directly using the
    `dreye.LogPhotoreceptor`, `dreye.LinearPhotoreceptor` and
    `dreye.HyperbolicPhotoreceptor` classes.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """
    if photoreceptor_type not in {'linear', 'log', 'hyperbolic', 'contrast'}:
        raise DreyeError("Photoreceptor type must be 'linear', 'log', 'hyperbolixc', or 'contrast'.")
    if photoreceptor_type == 'linear':
        pr_class = LinearPhotoreceptor
    elif photoreceptor_type == 'log':
        pr_class = LogPhotoreceptor
    elif photoreceptor_type == 'hyperbolic':
        pr_class = HyperbolicPhotoreceptor
    elif photoreceptor_type == 'contrast':
        pr_class = LinearContrastPhotoreceptor

    if is_signallike(sensitivity):
        pass
    elif (hasattr(sensitivity, 'ndim') and sensitivity.ndim > 1):
        if wavelengths is None:
            wavelengths = np.linspace(300, 700, len(sensitivity))
    elif is_listlike(sensitivity) or is_numeric(sensitivity):
        if is_integer(sensitivity) and (sensitivity < 100):  # reasonable threshold for sensitivity vs number of photoreceptors
            centers = np.linspace(350, 550, sensitivity)
        else:
            centers = np.atleast_1d(sensitivity)
        if wavelengths is None:
            wavelengths = np.arange(
                np.min(centers)-WL_PEAK_SPACE, 
                np.max(centers)+WL_PEAK_SPACE, 
                1
            )
        sensitivity = govardovskii2000_template(wavelengths[:, None], centers[None, :])
    elif sensitivity is None:
        if wavelengths is None:
            wavelengths = np.arange(300, 701, 1)
        centers = np.linspace(350, 550, 3)
        sensitivity = govardovskii2000_template(wavelengths[:, None], centers[None, :])

    return pr_class(
        sensitivity,
        wavelengths=wavelengths,
        filterfunc=filterfunc,
        labels=labels,
        **kwargs
    )


# TODO convenience capture function
# def capture


class Photoreceptor(ABC):
    """
    Abstract Photoreceptor model for subclassing.

    Parameters
    ----------
    sensitivity : Signals instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        absolute captures as a lower bound value.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.create_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    When subclassing this abstract class you need to implement two functions,
    that are `excitefunc` and `inv_excitefunc`. The `excitefunc` method
    accepts a an array-like object that converts photon capture values to
    photoreceptor excitations. The `inv_excitefunc` method accepts
    photoreceptor excitations and converts them back to photon capture values.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    def __init__(
        self, sensitivity, wavelengths=None, filterfunc=None,
        labels=None, capture_noise_level=None, **kwargs
    ):
        if isinstance(sensitivity, Photoreceptor):
            if filterfunc is None:
                filterfunc = sensitivity.filterfunc
            if capture_noise_level is None:
                capture_noise_level = sensitivity.capture_noise_level
            sensitivity = sensitivity.sensitivity
        if not isinstance(sensitivity, Signals):
            domain_units = kwargs.pop('domain_units', 'nm')
            sensitivity = Signals(
                sensitivity, domain=wavelengths, labels=labels,
                domain_units=domain_units, 
                **kwargs
            )
        else:
            if wavelengths is not None:
                sensitivity = sensitivity(wavelengths)
            if labels is not None:
                sensitivity = sensitivity.copy()
                sensitivity.labels = labels

        # ensure domain axis = 0
        if sensitivity.domain_axis != 0:
            sensitivity = sensitivity.copy()
            sensitivity.domain_axis = 0

        assert is_callable(filterfunc) or filterfunc is None

        self._sensitivity = sensitivity
        self._filterfunc = filterfunc
        self._capture_noise_level = capture_noise_level

    @property
    def data(self):
        """
        `numpy.ndarray` of sensitivities.
        """
        return self.sensitivity.magnitude

    @property
    def wls(self):
        """
        `numpy.ndarray` of wavelengths
        """
        return self.sensitivity.domain.magnitude

    def sensitivity_ratios(self, eps=0.01):
        """
        Compute ratios of the sensitivities
        """
        s = self.data + eps
        if np.any(s < 0):
            warnings.warn(
                "Zeros or smaller in sensitivities array!", RuntimeWarning
            )
            s[s < 0] = 0
        return s / np.sum(np.abs(s), axis=1, keepdims=True)

    def wavelength_range(self, rtol=None, peak2peak=False):
        """
        Range of wavelengths that the photoreceptors are sensitive to.
        Returns a tuple of the min and max wavelength value.
        """
        if peak2peak:
            dmax = self.sensitivity.dmax
            return np.min(dmax), np.max(dmax)
        rtol = (RELATIVE_SENSITIVITY_SIGNIFICANT if rtol is None else rtol)
        tol = (
            (self.sensitivity.max() - self.sensitivity.min())
            * rtol
        )
        return self.sensitivity.nonzero_range(tol).boundaries

    def __str__(self):
        """
        String representation of photoreceptor model.
        """
        return f"{type(self).__name__}(\n\t{self.sensitivity.magnitude}\n)"

    def __repr__(self):
        """
        String representation of photoreceptor model.
        """
        return f"{type(self).__name__}{tuple(self.labels)}"

    def to_dict(self):
        """
        Returns `self` as dictionary containing the
        photoreceptor's sensitivity and the filter function.
        """
        return {
            "sensitivity": self.sensitivity,
            "filterfunc": self.filterfunc,
            "capture_noise_level": self.capture_noise_level
        }

    @classmethod
    def from_dict(cls, dictionary):
        """
        Create Photoreceptor class given a dictionary containing
        the `sentivity` and `filterfunc` (optional) keys.
        """
        return cls(**dictionary)

    def __copy__(self):
        """
        Copy method for `Photoreceptor`.
        """
        return type(self)(
            **self.to_dict().copy()
        )

    def copy(self):
        """
        Copy `Photoreceptor` instance.
        """
        return copy.copy(self)

    def excitefunc(self, arr):
        """excitation function
        """
        return arr

    def inv_excitefunc(self, arr):
        """inverse of excitation function
        """
        return arr

    @property
    def sensitivity(self):
        """
        The `dreye.Sensitivity` instance containing
        all the spectral sensitivities.
        """
        return self._sensitivity

    @property
    def pr_number(self):
        """
        The number of photoreceptor types. This correspond to the number
        of spectral sensitivities of the `sensitivity` attribute.
        """
        return self.sensitivity.shape[1]

    @property
    def capture_noise_level(self):
        """
        Noise level for capture values
        """
        return (
            0 if ((self._capture_noise_level is None) or np.all(np.isnan(self._capture_noise_level)))
            else self._capture_noise_level
        )

    @property
    def wavelengths(self):
        """
        A `dreye.Domain` instance of the wavelength values.
        """
        return self.sensitivity.wavelengths

    @property
    def labels(self):
        """
        Labels of each photoreceptor type.

        See Also
        --------
        Photoreceptor.names
        """
        return self.sensitivity.labels

    @property
    def names(self):
        """
        Names of each photoreceptor type.

        Alias for `Photoreceptor.labels`.
        """
        return self.labels

    @property
    def filterfunc(self):
        """
        Filter function applied on light entering each photoreceptor.

        The filter function should accept three positional arguments
        corresponding to the wavelength, illuminant, and the sensitvity.
        All three are `numpy.ndarray` objects that have been broadcast
        properly to allow for mathematical operations.
        """
        return self._filterfunc

    def excitation(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None,
        add_noise=True,
        **kwargs
    ):
        """
        Calculate photoreceptor excitation.

        Parameters
        ----------
        illuminant : `Signals` or array-like
            Illuminant spectrum. If array-like, the zeroth axis must have
            the same length as `Photoreceptor.sensitivity` (i.e. wavelength
            domain match) or `wavelengths` must be given.
        reflectance : `Signals` or array-like, optional
            If given, illuminant is multiplied by the reflectance.
        background : `Signal` or array-like, optional
            If given, the relative photon capture is calculated. By
            calculating the capture for the background.
        return_units : bool, optional
            If True, return a `pint.Quantity`. Otherwise, return a
            `numpy.ndarray`.
        wavelengths : array-like, optional
            If given and illuminant is not a `Signals` instance, this
            corresponds to its wavelength values.
        add_noise : bool, optional
            Whether to apply noise thresholding or not. Defaults to True.
        kwargs : dict
            Keyword arguments passed to the `Photoreceptor.excitefunc`
            function.

        Returns
        -------
        excitations : `numpy.ndarray` or `pint.Quantity`
            Array of row length equal to the number of illuminant and
            column length equal to the number of photoreceptor types.
        """
        return self.excitefunc(
            self.capture(illuminant=illuminant,
                         reflectance=reflectance,
                         background=background,
                         return_units=return_units,
                         wavelengths=wavelengths,
                         add_noise=add_noise),
            **kwargs
        )

    def capture(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None,
        add_noise=True
    ):
        """
        Calculate photoreceptor capture.

        Parameters
        ----------
        illuminant : `Signals` or array-like (wavelengths x samples)
            Illuminant spectrum. If array-like, the zeroth axis must have
            the same length as `Photoreceptor.sensitivity` (i.e. wavelength
            domain match) or `wavelengths` must be given.
        reflectance : `Signals` or array-like, optional
            If given, illuminant is multiplied by the reflectance.
        background : `Signal` or array-like, optional
            If given, the relative photon capture is calculated. By
            calculating the capture for the background.
        return_units : bool, optional
            If True, return a `pint.Quantity`. Otherwise, return a
            `numpy.ndarray`.
        wavelengths : array-like, optional
            If given and illuminant is not a `Signals` instance, this
            corresponds to its wavelength values.
        add_noise : bool, optional
            Whether to apply noise thresholding or not. Defaults to True.

        Returns
        -------
        captures : `numpy.ndarray` or `pint.Quantity`  (samples x opsin)
            Array of row length equal to the number of illuminant and
            column length equal to the number of photoreceptor types.
        """
        if return_units is None:
            return_units = has_units(illuminant)

        if illuminant.ndim > 2:
            raise ValueError("Illuminant must be 1- or 2-dimensional.")

        if (
            hasattr(illuminant, 'equalize_domains')
            and hasattr(illuminant, 'domain_axis')
        ):
            if illuminant.ndim == 1:
                # creates a copy
                illuminant = Signals(illuminant)
                # set domain axis
                illuminant.domain_axis = 0

            if illuminant.domain_axis != 0:
                illuminant = illuminant.copy()
                illuminant.domain_axis = 0

            if reflectance is not None:
                illuminant = illuminant * reflectance

            sensitivity, illuminant = self.sensitivity.equalize_domains(
                illuminant
            )
        else:
            if not has_units(illuminant):
                illuminant = illuminant * ureg(None)

            if illuminant.ndim == 1:
                illuminant = illuminant[:, None]

            if wavelengths is not None:
                illuminant = Signals(
                    illuminant, domain=wavelengths, domain_units='nm'
                )

                if reflectance is not None:
                    illuminant = illuminant * reflectance

                sensitivity, illuminant = self.sensitivity.equalize_domains(
                    illuminant
                )

            else:
                if reflectance is not None:
                    illuminant = illuminant * reflectance
                sensitivity = self.sensitivity

        wls = sensitivity.domain

        new_units = illuminant.units * sensitivity.units * wls.units

        # TODO dealing with 1-dimensional properly
        # added_dim = illuminant.ndim - 1
        # (slice(None, None, None), ) * 2 + (None, ) * added_dim
        sensitivity = sensitivity.magnitude[:, None, :]
        illuminant = illuminant.magnitude[:, :, None]
        wls = wls.magnitude[:, None, None]

        # illuminant can be filtered after equalizing domains
        # TODO how to handle background (also filter?)
        if self.filterfunc is not None:
            illuminant = self.filterfunc(wls, illuminant, sensitivity)

        # calculate capture
        # illuminant x opsin via integral
        q = np.trapz(sensitivity * illuminant, wls, axis=0)
        if add_noise:
            q += self.capture_noise_level

        if np.any(q < 0):
            raise ValueError("Capture values calculated are below 0; "
                             "Make sure illuminant, reflectance, background "
                             "and sensitivities do not contain negative "
                             "values.")

        if return_units:
            q = q * new_units

        if background is None:
            return q

        # calculate relative capture
        q_bg = self.capture(
            background, return_units=return_units, wavelengths=wavelengths
        )
        # q_bg may have different units to q
        q = q / q_bg
        return q


@inherit_docstrings
class LinearPhotoreceptor(Photoreceptor):
    """
    A linear photoreceptor model

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        absolute captures as a lower bound value.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.create_photoreceptor_model
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the linear model, photon capture values correspond to photoreceptor
    excitations.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns `arr`.
        """
        return arr

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns `arr`.
        """
        return arr


@inherit_docstrings
class LogPhotoreceptor(Photoreceptor):
    """
    A logarithmic photoreceptor model

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        absolute captures as a lower bound value.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.create_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the logarithmic photoreceptor model the photoreceptor excitations
    correspond to the log of the photon captures.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns the log of `arr`.
        """
        return np.log(arr)

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the exp of `arr`.
        """
        return np.exp(arr)


@inherit_docstrings
class HyperbolicPhotoreceptor(Photoreceptor):
    """
    A hyperbolic photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        absolute captures as a lower bound value.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.create_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor

    Notes
    -----
    In the hyperbolic photoreceptor model, the photoreceptor excitations
    correspond to the hyperbolic transform of the photon captures:
    :math:`(q-1)/q`.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns the  `(arr - 1)/arr`.
        """
        return (arr - 1) / arr

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the `1/(1-arr)`.
        """
        return 1 / (1 - arr)


@inherit_docstrings
class LinearContrastPhotoreceptor(Photoreceptor):
    """
    A linear contrast photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        absolute captures as a lower bound value.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.create_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the linear photoreceptor model, the photoreceptor excitations
    correspond to the fractional difference of the photon captures:
    :math:`q-1`
    with :math:`q` corresponding to the relative capture.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns the  `arr`/(1+`arr`).
        """
        return arr - 1

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the `arr`/(1 - `arr`).
        """
        return arr + 1
