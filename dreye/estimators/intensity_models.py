"""
"""

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    optional_to, asarray
)
from dreye.constants import ureg
from dreye.estimators.base import _SpectraModel


class IntensityFit(_SpectraModel):
    """
    Fit intensity values to a given LED system.

    All models that fit chromatic stimuli have to accept a
    measured spectra object.
    """

    # other attributes that are the length of X but not X
    _X_length = []

    def __init__(
        self,
        *,
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None  # float
    ):
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window

    def fit(self, X, y=None):
        """
        Fit method.
        """
        #
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra, self.smoothing_window, asarray(X).shape[1]
        )
        # check X
        X = self._check_X(X)
        self.current_X_ = X
        # call in order to fit isotonic regression
        self.measured_spectra_.regressor

        self.n_features_ = len(self.measured_spectra_)

        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from number"
                             "of measured spectra in container.")

        self.fitted_intensities_ = X

        return self

    def inverse_transform(self, X):
        """
        Inverse Transform method
        """
        # check is fitted
        check_is_fitted(self, ['n_features_', 'measured_spectra_'])

        # check X
        X = optional_to(X, self.output_units)
        X = check_array(X)

        # map output values to intensities
        return self.measured_spectra_.inverse_map(X, return_units=False)

    @property
    def input_units(self):
        """units of X
        """
        return self.measured_spectra_.intensities.units

    @property
    def fitted_X(self):
        """X after fitting
        """
        return self.fitted_intensities_


class RelativeIntensityFit(_SpectraModel):
    """
    Fit relative intensity values [log](I/I_0) to a given LED system.

    All models that fit chromatic stimuli have to accept a
    measured spectra object.
    """

    # other attributes that are the length of X but not X
    _X_length = [
        'fitted_intensities_'
    ]

    def __init__(
        self,
        *,
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        bg_ints=None,  # array-like
        smoothing_window=None,  # float
        rtype=None,  # {'fechner/log', 'weber', None}
    ):
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.rtype = rtype
        self.bg_ints = bg_ints

    def fit(self, X, y=None):
        """
        Fit method.
        """
        #
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra,
            self.smoothing_window,
            asarray(X).shape[1]
        )
        if self.bg_ints is None:
            self.bg_ints_ = np.ones(len(self.measured_spectra_))
        else:
            self.bg_ints_ = optional_to(
                self.bg_ints,
                self.measured_spectra_.intensities.units
            )
            assert len(self.bg_ints_) == len(self.measured_spectra_)
            assert np.all(self.bg_ints_ > 0)
        # check X
        X = self._check_X(X)
        self.current_X_ = X
        self.fitted_relative_intensities_ = X

        # call in order to fit isotonic regression
        self.measured_spectra_.regressor

        self.n_features_ = len(self.measured_spectra_)

        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from number"
                             "of measured spectra in container.")

        self.fitted_intensities_ = self._to_absolute_intensity(X)

        return self

    def _to_absolute_intensity(self, X):
        # convert to intensity
        if self.rtype in {'fechner', 'log'}:
            X = np.exp(X)
        elif self.rtype != 'weber':
            assert np.all(X >= 0), 'If not log, X must be positive.'

        X = X * self.bg_ints_[None]
        if self.rtype == 'weber':
            X = X + self.bg_ints_[None]
        return X

    def _to_relative_intensity(self, X):
        # convert to relative intensity
        if self.rtype == 'weber':
            X = X - self.bg_ints_[None]

        X = X / self.bg_ints_[None]

        assert np.all(X >= 0)

        if self.rtype in {'fechner', 'log'}:
            # may have infs!
            X = np.log(X)

        return X

    def inverse_transform(self, X):
        """
        Inverse Transform method
        """
        # check is fitted
        check_is_fitted(self, ['n_features_', 'measured_spectra_'])

        # check X
        X = optional_to(X, self.output_units)
        X = check_array(X)

        # map output values to intensities
        return self._to_relative_intensity(
            self.measured_spectra_.inverse_map(X, return_units=False)
        )

    @property
    def input_units(self):
        """units of X
        """
        return ureg(None).units

    @property
    def fitted_X(self):
        """X after fitting
        """
        return self.fitted_relative_intensities_

#
# # TODO improve (not final version)
# class IlluminantFit(_SpectraModel):
#     """
#     Fit illuminant spectra
#     """
#
#     def __init__(
#         self,
#         *,
#         measured_spectra=None,  # dict, or MeasuredSpectraContainer
#         smoothing_window=None,  # float
#         max_iter=None
#     ):
#         self.measured_spectra = measured_spectra
#         self.smoothing_window = smoothing_window
#         self.max_iter = max_iter
#
#     def fit(self, X, y=None):
#         """
#         Fit method.
#         """
#         # create measured_spectra_
#         self.measured_spectra_ = self._check_measured_spectra(
#             self.measured_spectra, self.smoothing_window
#         )
#         normalized_spectra = self.measured_spectra_.normalized_spectra.copy()
#         assert normalized_spectra.domain_axis == 0
#         # make 2D if necessary
#         if isinstance(X, Signal):
#             X = Signals(X)
#         # move domain axis and equalize if necessary
#         if isinstance(X, _Signal2DMixin):
#             X = X.copy()
#             # ensure domain axis is feature axis
#             X.domain_axis = 1
#             normalized_spectra, X = normalized_spectra.equalize_domains(X)
#         # check X
#         X = self._check_X(X)
#         # also store checked X
#         self.current_X_ = X
#         # spectra as array
#         self.normalized_spectra_ = normalized_spectra
#         self.wavelengths_ = self.normalized_spectra_.domain.magnitude
#         self.bounds_ = self.measured_spectra_.intensity_bounds
#
#         # creates regressor for mapping values
#         self.measured_spectra_.regressor
#         self.container_ = self._fit_samples(X)
#
#         if not np.all(self.container_.success):
#             warnings.warn("Convergence was not accomplished "
#                           "for all spectra in X; "
#                           "increase the number of max iterations.")
#
#         self.n_features_ = X.shape[1]
#         # samples x intensity
#         self.fitted_intensities_ = np.array(
#             self.container_.x
#         ) * self.measured_spectra_.intensities.units
#         # or self.input_units / self.normalized_spectra_.units
#
#         return self
#
#     def _fit_samples(self, X):
#         """
#         Fit individual samples
#         """
#         # TODO accuracy for wavelength range e.g. 10nm (in blocks)
#         # TODO first integrate window filter?
#         A = asarray(self.normalized_spectra_)
#         container = OptimizeResultContainer()
#         for x in X:
#             container.append(
#                 lsq_linear(
#                     A, x,
#                     bounds=tuple(self.bounds_),
#                     max_iter=self.max_iter
#                 )
#             )
#         return container
#
#     def inverse_transform(self, X):
#         """
#         Transform output values to spectra
#         """
#         check_is_fitted(
#             self, ['measured_spectra_', 'normalized_spectra_']
#         )
#         # X is samples x LEDs
#         X = optional_to(X, self.output_units)
#         X = check_array(X)
#
#         assert X.shape[1] == len(self.measured_spectra_)
#
#         # samples x LED
#         X = self.measured_spectra_.inverse_map(X, return_units=False)
#         return X @ self.normalized_spectra_.magnitude.T
#
#     @property
#     def input_units(self):
#         """units of X
#         """
#         return self.measured_spectra_.units