"""
LED Estimators for intensities and spectra
"""

from abc import abstractmethod
import warnings

from scipy.optimize import OptimizeResult
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import optional_to
from dreye.utilities.abstract import _AbstractContainer
from dreye.estimators.utils import (
    check_background, check_measured_spectra, 
    check_photoreceptor_model, 
    estimate_bg_ints_from_background, get_background_from_bg_ints, 
    get_bg_ints, get_ignore_bounds
)
from dreye.utilities.common import is_string


class OptimizeResultContainer(_AbstractContainer):
    _allowed_instances = OptimizeResult


class _SpectraModel(BaseEstimator, TransformerMixin):
    """
    Abstract Spectra model used for various Dreye estimators.
    """
    _deprecated_kws = {
        "smoothing_window": None
    }

    # other attributes that are the length of X but not X
    # and are not the fitted signal
    _X_length = []

    def fit_transform(self, X):
        """
        Fit X values and transform to output values
        of `MeasuredSpectraContainer`.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        output : array-like (n_samples, n_leds)
            The transformed X values into LED output values.

        See Also
        --------
        fit
        transform
        """
        self.fit(X)
        return self.transform()

    def _check_X(self, X):
        """units for X.

        e.g. in photonflux units.
        """
        X = optional_to(X, self.input_units)
        return check_array(X)

    @property
    def output_units(self):
        """
        Units of transformed X.
        """
        return self.measured_spectra_.labels_units

    @property
    @abstractmethod
    def input_units(self):
        """
        Units of X.
        """

    @property
    @abstractmethod
    def fitted_X_(self):
        """
        X after fitting.
        """

    @property
    @abstractmethod
    def X_(self):
        """
        X before fitting.
        """

    def to_dict(self):
        """
        Get estimator initial dictionary.

        See Also
        --------
        from_dict
        """
        return self.get_params()

    @classmethod
    def from_dict(cls, data):
        """
        Create estimator from intial dictionary

        See Also
        --------
        to_dict
        """
        return cls(**data)

    def residuals(self, X=None):
        """
        Residual for each photoreceptor and sample.
        """
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.X_
        return X - X_pred

    def relative_changes(self, X=None):
        """
        Deviations for each photoreceptor and sample.
        """
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.X_
        return (X_pred - X) / np.abs(X)

    @staticmethod
    def _r2_scores(X, X_pred, axis=0):
        # residual across photoreceptors
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=axis, keepdims=True))**2
        return 1 - res.sum(axis) / tot.sum(axis)

    @staticmethod
    def _agg_scores(scores, axis, aggfunc=None, default='mean'):
        if aggfunc is None:
            aggfunc = default
        if isinstance(aggfunc, str):
            aggfunc = getattr(np, aggfunc)
        return aggfunc(scores, axis=axis)

    def _abs_rel_changes_scores(self, X, X_pred, axis=0, round=None, aggfunc=None):
        X, X_pred = _round_args(X, X_pred, round=round)
        # absolute deviations
        scores = np.abs((X_pred - X) / X)
        return self._agg_scores(scores, axis, aggfunc)

    def _rel_changes_scores(self, X, X_pred, axis=0, round=None, aggfunc=None):
        X, X_pred = _round_args(X, X_pred, round=round)
        # deviations
        scores = ((X_pred - X) / np.abs(X))
        return self._agg_scores(scores, axis, aggfunc)

    def _rel_changes_thresh_scores(self, X, X_pred, axis=0, thresh=0.01, round=None, aggfunc=None):
        X, X_pred = _round_args(X, X_pred, round=round)
        # absolute deviations below a certain threshold value
        scores = (np.abs((X_pred - X) / X) < thresh)
        return self._agg_scores(scores, axis, aggfunc)

    def _corr_dist(self, X, X_pred, axis=0, round=None, aggfunc=None):
        # correlation distortion
        cX = np.corrcoef(X, rowvar=bool(axis % 2))
        cX_pred = np.corrcoef(X_pred, rowvar=bool(axis % 2))
        cX, cX_pred = _round_args(cX, cX_pred, round=round)
        scores = (cX_pred - cX) / np.abs(cX)

        def default(x, axis):
            return x

        return self._agg_scores(scores, axis, aggfunc, default=default)

    def _mean_scores(self, X=None, axis=0, method='r2', **kwargs):
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.X_

        # residual across photoreceptors
        if method == 'r2':
            return self._r2_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'rel':
            return self._rel_changes_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'absrel':
            return self._abs_rel_changes_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'threshrel':
            return self._rel_changes_thresh_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'corrdev':
            return self._corr_dist(
                X, X_pred, axis=axis, **kwargs
            )
        elif callable(method):
            return method(
                X, X_pred, axis=axis, **kwargs
            )
        # raise
        raise NameError(f"Method `{method}` not recognized, use "
                        "`r2`, `rel`, `absrel`, `threshrel`, or `corrdev`.")

    def sample_scores(self, X=None, method='r2', **kwargs):
        """
        Sample scores.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : numpy.ndarray (n_samples)
            Returns the r2-value for each sample individually.
        """
        return self._mean_scores(X=X, axis=1, method=method, **kwargs)

    def feature_scores(self, X=None, method='r2', **kwargs):
        """
        Feature scores.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : numpy.ndarray (n_features)
            Returns the r2-value for each feature individually.
        """
        return self._mean_scores(X=X, axis=0, method=method, **kwargs)

    def score(self, X=None, y=None, method='r2', **kwargs):
        """
        Score method.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : float
            Mean r2-value across features

        See Also
        --------
        feature_scores
        """
        return self.feature_scores(X, method=method, **kwargs).mean()

    def transform(self, X=None):
        """
        Transform X values to output values of `MeasuredSpectraContainer`.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        output : array-like (n_samples, n_leds)
            The transformed X values into LED output values.

        See Also
        --------
        dreye.MeasuredSpectraContainer.map
        dreye.MeasuredSpectrum.map
        fit_transform
        """
        # check is fitted
        check_is_fitted(
            self, ['measured_spectra_', 'fitted_intensities_']
        )

        if X is not None:
            # refit if X is given
            self.fit(X)
        # map fitted_intensities
        return self.measured_spectra_.map(
            self.fitted_intensities_, 
            return_units=False, 
            check_bounds=not get_ignore_bounds(
                self.ignore_bounds, self.measured_spectra, self.intensity_bounds
            )
        )

    @abstractmethod
    def inverse_transform(self, X):
        """
        Transform output values to values that can be used for scoring.

        Parameters
        ----------
        X : array-like (n_samples, n_leds)
            Output values in units of `output_units`.

        Returns
        -------
        X : array-like (n_samples, n_features)
            Input values in units of `input_units`.

        See Also
        --------
        dreye.MeasuredSpectraContainer.inverse_map
        dreye.MeasuredSpectrum.inverse_map
        fit_transform
        """
        pass

    @abstractmethod
    def fit(self, X):
        """
        Fit X values.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values in `input_units`.

        Returns
        -------
        self

        See Also
        --------
        dreye.MeasuredSpectraContainer
        dreye.MeasuredSpectrum
        fit_transform
        """
        pass


class _RelativeMixin:
    # requires setting bg_ints_
    rtype = None

    def _to_absolute_intensity(self, X, bg_ints=None):
        assert self.rtype is not None, "`rtype` cannot be NoneType."
        if bg_ints is None:
            bg_ints = self.bg_ints_
        if self.rtype == 'absolute':
            return X
        elif self.rtype == 'diff':
            return X + bg_ints
        # convert to intensity
        if self.rtype in {'fechner', 'log'}:
            X = np.exp(X)
        elif self.rtype not in {'weber', 'total_weber'}:
            assert np.all(X >= 0), 'If not log, X must be positive.'

        if self.rtype.startswith('total'):
            X = X * np.sum(self.bg_ints_)
        else:
            X = X * bg_ints
        if self.rtype in {'weber', 'total_weber'}:
            X = X + bg_ints
        return X

    def _to_relative_intensity(self, X, bg_ints=None):
        assert self.rtype is not None, "`rtype` cannot be NoneType."
        if bg_ints is None:
            bg_ints = self.bg_ints_
        if self.rtype == 'absolute':
            return X
        elif self.rtype == 'diff':
            return X - bg_ints
        # convert to relative intensity
        if self.rtype in {'weber', 'total_weber'}:
            X = X - bg_ints

        if self.rtype.startswith('total'):
            X = X / np.sum(self.bg_ints_)
        else:
            X = X / bg_ints

        if self.rtype in {'fechner', 'log'}:
            assert np.all(X > 0), 'If log, X cannot be zero or lower.'
            X = np.log(X)
        elif self.rtype not in {'weber', 'total_weber'}:
            assert np.all(X >= 0), 'If not log, X must be positive.'

        return X


class _PrModelMixin:
    _lazy_background_estimation = False
    # required attributes
    ignore_bounds = None
    measured_spectra = None
    photoreceptor_model = None
    wavelengths = None
    intensity_bounds = None
    background = None
    bg_ints = None
    background_external = None
    capture_noise_level = None
    """
    The following attributes will be set (using `_set_pr_model_related_objects` method):
    * photoreceptor_model_
    * measured_spectra_
    * noise_term_
    * A_
    * q_bg_
    * intensity_bounds_
    * background_
    * bg_ints_

    Adds methods:
    * get_capture
    * get_excitation
    """

    def _set_photoreceptor_model_and_measured_spectra_object(self, size=None):
        # create photoreceptor model
        self.photoreceptor_model_ = check_photoreceptor_model(
            self.photoreceptor_model, size=size, 
            wavelengths=self.wavelengths, 
            capture_noise_level=self.capture_noise_level
        )

        # create measured_spectra_
        self.measured_spectra_ = check_measured_spectra(
            self.measured_spectra,
            photoreceptor_model=self.photoreceptor_model_, 
            wavelengths=self.wavelengths, 
            intensity_bounds=self.intensity_bounds
        )

    def _set_A(self):
        if np.any(self.photoreceptor_model_.capture_noise_level):
            self.noise_term_ = self.photoreceptor_model_.capture(
                np.zeros(self.measured_spectra_.normalized_spectra.domain.size),
                wavelengths=self.measured_spectra_.normalized_spectra.domain,
                background=self.background_, 
                return_units=False
            )
        else:
            self.noise_term_ = 0

        # opsin x LED (taking transpose)
        self.A_ = (
            self.photoreceptor_model_.capture(
                self.measured_spectra_.normalized_spectra,
                background=self.background_,
                return_units=False
            ) - self.noise_term_
        ).T

        # capture from a background light source (not in measured_spectra)
        if self._background_external_:
            # q_bg_ has the noise term removed
            # 1d array
            self.q_bg_ = self.photoreceptor_model_.capture(
                self.background_ - self._bg_ints_background_,
                background=self.background_,
                return_units=False
            ) - self.noise_term_ # length of opsin
        else:
            # no noise term to remove if all the 
            # light is coming from the sources
            self.q_bg_ = 0

        return self

    def _set_intensity_bounds(self):
        ignore_bounds = get_ignore_bounds(
            self.ignore_bounds, self.measured_spectra, self.intensity_bounds
        )
        # measured_spectra attributes
        # intensity bounds as two-tuple
        if ignore_bounds:
            self.intensity_bounds_ = (
                np.zeros(len(self.measured_spectra_)),
                np.inf * np.ones(len(self.measured_spectra_))
            )
        elif self.intensity_bounds is not None:
            self.intensity_bounds_ = self.intensity_bounds
        else:
            self.intensity_bounds_ = self.measured_spectra_.intensity_bounds

    def _set_background(self):
        """
        set `background_` and `bg_ints_`

        also sets internals _background_external_ and _bg_ints_background_
        """
        # create background
        self.background_ = check_background(
            self.background, self.measured_spectra_, 
            wavelengths=self.wavelengths
        )

        # set background intensities if exist
        self.bg_ints_ = get_bg_ints(
            self.bg_ints, self.measured_spectra_,
            rtype=getattr(self, 'rtype', None)
        )

        if self.background_ is None:
            self._background_external_ = False
            self._bg_ints_background_ = get_background_from_bg_ints(
                self.bg_ints_, self.measured_spectra_
            )
            
            if self.background_external:
                warnings.warn(
                    "Ignoring `background_external` argument, "
                    "automatically set to False as `background` is None."
                )

            if (
                not np.allclose(self.bg_ints_, 0) 
                or np.all(self.photoreceptor_model_.capture_noise_level)
            ):
                # will integrate with normalized spectra
                # all have noise - then do relative q
                self.background_ = self._bg_ints_background_
                # otherwise keep background_ None

        elif is_string(self.background_) and (self.background_ in {'mean', 'norm'}):
            self._background_external_ = False
            self._bg_ints_background_ = 0
            assert np.allclose(self.bg_ints_, 0), "If background is `{self.background_}`, bg_ints must be all zeros."

        elif self.background_external or self.background_external is None:
            # if background is not None assume it is from an external source (default)
            self._background_external_ = True
            self._bg_ints_background_ = get_background_from_bg_ints(
                self.bg_ints_, self.measured_spectra_
            )
            self.background_ = self.background_ + self._bg_ints_background_

        # if background and bg_ints were given, and background_external was set to False
        # check that things match accordingly
        # NB: have to use bg_ints as bg_ints_ is never None except for lazy_background_estimation
        elif self.bg_ints is not None:
            self._background_external_ = False
            self._bg_ints_background_ = get_background_from_bg_ints(
                self.bg_ints_, self.measured_spectra_
            )

            if not np.allclose(
                self.background_.magnitude, self._bg_ints_background_.magnitude
            ):
                raise ValueError(
                    "The provided `background` and `bg_ints` do not match even though "
                    "`background_external` was set to False." 
                )

        elif self._lazy_background_estimation:
            # set bg_ints_ to None for lazy background estimation
            self.bg_ints_ = None
            self._background_external_ = False
            self._bg_ints_background_ = 0

        # Not having run estimation process of background intensities before
        # background_external was set to False
        else:
            self._background_external_ = False
            self._bg_ints_background_ = 0
            Xbg = np.ones(self.photoreceptor_model_.n_opsins)
            Xbg = self.photoreceptor_model_.excitefunc(Xbg)
            self.bg_ints_ = estimate_bg_ints_from_background(
                Xbg,
                self.photoreceptor_model,
                self.background, 
                self.measured_spectra, 
                fit_weights=getattr(self, 'fit_weights', None),
                max_iter=getattr(self, 'max_iter', None),
                ignore_bounds=getattr(self, 'ignore_bounds', None),
                lsq_kwargs=getattr(self, 'lsq_kwargs', None),
                intensity_bounds=self.intensity_bounds, 
                wavelengths=self.wavelengths
            )
            warnings.warn(
                "Assuming the `background` illuminant will be simulated "
                f"using the LEDs. Fitted background intensities lazily: "
                f"{self.bg_ints_.tolist()}."
            )

        # sanity
        if (
            not (is_string(self.background_) or self.background_ is None)
            and np.allclose(self.background_.magnitude, 0)
            and not np.all(self.photoreceptor_model_.capture_noise_level)  # not all are noisy - then don't keep relative q
        ):
            warnings.warn(
                "background array is all zero and no capture noise, "
                "setting `background` illuminant to None."
            )
            self.background_ = None

    def _set_pr_model_related_objects(self, size=None):
        # set photoreceptor_model_ and measured_spectra_
        self._set_photoreceptor_model_and_measured_spectra_object(size)

        # set intensity_bounds_
        self._set_intensity_bounds()

        # sets background_, bg_ints_, and internals
        self._set_background()

        # set A_, noise_term_, and q_bg_
        self._set_A()

    def get_capture(self, w):
        """
        Get capture given `w`.

        Parameters
        ----------
        w : array-like
            Array-like object with the zeroth axes equal to the number of light sources. 
            Can also be multidimensional.
        """
        if self.photoreceptor_model_.filterfunc is None:
            # threshold by noise if necessary and apply nonlinearity
            x_pred = (self.A_ @ w).T
            if np.any(self.photoreceptor_model_.capture_noise_level):
                x_pred += self.noise_term_
        else:
            warnings.warn("Fitting with filterfunc not tested!", RuntimeWarning)
            illuminant = self.measured_spectra_.ints_to_spectra(w)
            x_pred = self.photoreceptor_model_.capture(
                illuminant,
                background=self.background_,
                return_units=False
            )
        x_pred += self.q_bg_
        return x_pred

    def get_excitation(self, w):
        """
        Get excitation given `w`.

        Parameters
        ----------
        w : array-like
            Array-like object with the zeroth axes equal to the number of light sources. 
            Can also be multidimensional.
        """
        return self.photoreceptor_model_.excitefunc(self.get_capture(w))


# -- misc helper functions


def _round_args(*args, round=None):
    if round is not None:
        return tuple([np.round(arg, round) for arg in args])
    return args
