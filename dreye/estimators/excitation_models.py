"""
Excitation models
"""

import warnings

import numpy as np
import cvxpy as cp
from scipy.optimize import OptimizeResult
from scipy.optimize import lsq_linear, least_squares
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    optional_to, asarray, is_listlike, is_callable
)
from dreye.utilities.common import is_numeric, is_string
from dreye.constants import ureg
from dreye.estimators.base import _SpectraModel, _PrModelMixin, OptimizeResultContainer
from dreye.err import DreyeError
from dreye.utilities.abstract import inherit_docstrings


# TODO what if people want to supply q_bg instead of background - could just scale the sensitivities


@inherit_docstrings
class IndependentExcitationFit(_SpectraModel, _PrModelMixin):
    """
    Class to fit (relative) photoreceptor excitations for each sample
    independently.

    Parameters
    ----------
    photoreceptor_model : dreye.Photoreceptor, optional
        A photoreceptor model that implements the `capture`, `excitation`,
        `excitefunc` and `inv_excitefunc` methods. If None,
        a fake photoreceptor model will be created with three different
        photoreceptor types.
    fit_weights : array-like, optional
        Weighting of the importance of each photoreceptor type in the model.
        If None, weighting will be equal between all photoreceptor types.
        Must be same length as the number of photoreceptor types
    background : dreye.Signal, optional
        The spectral distribution of the background illuminant. Assumes
        that it contains the whole added spectrum including the LED intensities,
        if they are non-zero.
    measured_spectra : dreye.MeasuredSpectraContainer, optional
        Container with all available LEDs and their measured spectra. If
        None, a fake LED measurement will be created with intensities
        ranging from 0 to 100 microphotonflux.
    max_iter : int, optional
        The number of maximum iterations. This is passed directly to
        `scipy.optimize.lsq_linear` and `scipy.optimize.least_squares`.
    bg_ints : numpy.ndarray, optional
        The intensity values for each LED, when the relative capture of each
        photoreceptor equals one (i.e. background intensity).
        This will prevent fitting of the
        LED intensities if the background LED intensities
        are preset and the relative capture is 1.
    fit_only_uniques : bool, optional
        If True, use `numpy.unique` to select only the unique samples
        for fitting before transforming X back to the full array.
    ignore_bounds : bool, optional
        If True, ignore the bounds of the LED intensities. Howerver, a zero
        LED intensity bound will always exist.
    lsq_kwargs : dict, optional
        Keyword arguments passed directly to `scipy.optimize.least_squares`.

    Attributes
    ----------
    photoreceptor_model_ : dreye.Photoreceptor
        The photoreceptor model used for fitting. This will be the same if
        as `photoreceptor_model` if it is a `dreye.Photoreceptor` instance.
    measured_spectra_ : dreye.MeasuredSpectraContainer (n_leds)
        Measured spectrum container used for fitting. This will be the same
        if as `measured_spectra` if a `dreye.MeasuredSpectraContainer` instance
        was passed.
    intensity_bounds_ : numpy.ndarray (n_leds)
        The LED intensity bounds used for fitting.
    background_ : dreye.Spectrum
        The background used for calculating the relative photon capture.
    A_ : numpy.ndarray (n_prs, n_leds)
        The relative photon capture of each normalized LED spectrum.
    capture_X_ : numpy.ndarray (n_samples, n_prs)
        The current relative photon capture values used for fitting.
    excite_X_ : numpy.ndarray (n_samples, n_prs)
        The current photoreceptor excitation values used for fitting.
    fitted_intensities_ : numpy.ndarray
        Intensities fit in units of `measured_spectra_.intensities.units`
    fitted_capture_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated relative photon capture values after fitting.
    fitted_excite_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated photoreceptor excitations after fitting.
    """

    # same length as X but not X or fitted X
    _X_length = [
        'fitted_intensities_'
    ]
    _deprecated_kws = {
        **_SpectraModel._deprecated_kws,
        "photoreceptor_fit_weights": "fit_weights",
        "q1_ints": "bg_ints",
        "smoothing_window": None, 
        "hard_separation": "unidirectional", 
        "hard_sep_value": None, 
        "ignore_capture_units": None, 
        "background_only_external": None
    }
    _fit_to_transform = False
    _requirements_set = False

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        unidirectional=False,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        lsq_kwargs=None,
        background_external=None, 
        intensity_bounds=None, 
        wavelengths=None, 
        capture_noise_level=None, 
        underdetermined_opt=None
    ):
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.background = background
        self.max_iter = max_iter
        self.unidirectional = unidirectional
        self.fit_weights = fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs
        self.ignore_bounds = ignore_bounds
        self.bg_ints = bg_ints
        self.background_external = background_external
        self.intensity_bounds = intensity_bounds
        self.wavelengths = wavelengths
        self.capture_noise_level = capture_noise_level
        self.underdetermined_opt = underdetermined_opt

    def _set_required_objects(self, size=None):
        # if requirements set do not reset them (speed improvement)
        if self._requirements_set:
            return self

        # set photoreceptor_model_ and measured_spectra_
        # set intensity_bounds_
        # set background_, bg_ints_, and internals
        # set A_, noise_term_, and q_bg_
        self._set_pr_model_related_objects(size)

        # number of photoreceptors
        # if background is None then capture border is 0
        self.capture_border_ = float(self.background_ is not None)

        # weighting for each photoreceptor
        if self.fit_weights is None:
            self.fit_weights_ = np.ones(self.photoreceptor_model_.n_opsins)
        else:
            self.fit_weights_ = asarray(self.fit_weights)
            # assert len(fit_weights) == self.photoreceptor_model_.n_opsins

        # do weighted initial fitting in linear domain
        self.weighted_init_fit_ = (
            not self._fit_to_transform
            and np.unique(self.fit_weights_).size > 1
        )

        # set _unidirectional_
        self._set_unidirectional()

        # make sure requirements are only set once
        self._requirements_set = True

        return self

    def _set_unidirectional(self):
        if is_listlike(self.unidirectional):
            warnings.warn(
                "List-like `unidirectional` type is deprecated, and will be set to `True`.", 
                DeprecationWarning
            )
            self._unidirectional_ = True
        elif self.unidirectional and (self.bg_ints_ is None):
            warnings.warn(
                "`unidirectional` set to True, but `bg_ints` is None; "
                "`unidirectional` will be set to `False`.", 
                RuntimeWarning
            )
            self._unidirectional_ = False
        else:
            self._unidirectional_ = self.unidirectional

    def _fit(self, X):
        """
        Actual Fitting method. Allows easier subclassing
        """

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)

        # if only fit uniques used different iterator
        if self.fit_only_uniques:
            # get uniques
            _, xidcs, xinverse = np.unique(
                self.capture_X_, axis=0, return_index=True, return_inverse=True
            )
            self.container_ = OptimizeResultContainer(np.array([
                self._fit_sample(
                    capture_x, excite_x
                )
                for capture_x, excite_x in
                zip(self.capture_X_[xidcs], self.excite_X_[xidcs])
            ])[xinverse])
        else:
            self.container_ = OptimizeResultContainer([
                self._fit_sample(
                    capture_x, excite_x
                )
                for capture_x, excite_x in
                zip(self.capture_X_, self.excite_X_)
            ])

        if not np.all(self.container_.success):
            warnings.warn("Convergence was not accomplished "
                          "for all spectra in X; "
                          "increase the number of max iterations.", RuntimeWarning)

        self.fitted_intensities_ = np.array(self.container_.x)
        self.optimal_ = np.array(self.container_.status) >= 4
        # get fitted X
        self.fitted_excite_X_ = self.get_excitation(self.fitted_intensities_.T)
        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )

        return self

    def fit(self, X, y=None):
        # set required objects
        self._set_required_objects(asarray(X).shape[1])
        # fit X
        self._fit(X)
        return self

    def inverse_transform(self, X):
        check_is_fitted(
            self, [
                'measured_spectra_',
                'photoreceptor_model_',
                'background_'
            ]
        )
        # X is samples x LEDs
        X = optional_to(X, self.output_units)
        X = check_array(X)

        assert X.shape[1] == len(self.measured_spectra_)

        # got from output to intensity
        X = self.measured_spectra_.inverse_map(X, return_units=False)
        # get excitation given intensity
        # n_sources x samples passed returns samples x n_opsins
        X = self.get_excitation(X.T)
        return X

    def _fit_sample(self, capture_x, excite_x):
        # adjust bounds if necessary
        bounds = list(self.intensity_bounds_)
        if self._unidirectional_:
            if np.all(capture_x >= self.capture_border_):
                bounds[0] = self.bg_ints_
            elif np.all(capture_x <= self.capture_border_):
                bounds[1] = self.bg_ints_
        if self.bg_ints_ is not None:  # None only when doing lazy_background_estimation
            # if np.allclose(capture_x, 1)
            # if close to the background intensity just use those values
            if np.allclose(capture_x, self.capture_border_):
                return OptimizeResult(
                    x=self.bg_ints_,
                    cost=0.0,
                    fun=np.zeros(capture_x.size),
                    jac=np.zeros((capture_x.size, self.bg_ints_.size)),
                    grad=np.zeros(capture_x.size),
                    optimality=0.0,
                    active_mask=0,
                    nfev=1,
                    njev=None,
                    status=4,
                    message='Set result to background intensities.',
                    success=True
                )
        # find initial w0 using linear least squares
        result = self._init_sample(capture_x, bounds, return_result=True)
        # non-perfect solution
        if (
            (self.underdetermined_opt is None) 
            # non-perfect solution not found
            or (result.status != 3) 
            # not underdetermined
            or (self.A_.shape[0] >= self.A_.shape[1])
        ):
            # fit result via nonlinear least squares
            return least_squares(
                self._objective,
                x0=result.x,
                args=(excite_x,),
                bounds=tuple(bounds),
                max_nfev=self.max_iter,
                **({} if self.lsq_kwargs is None else self.lsq_kwargs)
            )
        else:
            # underdetermined and optimal system
            w = cp.Variable(self.A_.shape[1], pos=True)
            constraints = [
                self.get_capture(w) == capture_x, 
                w >= bounds[0], w <= bounds[1]
            ]

            if is_numeric(self.underdetermined_opt):
                # aim for some overall intensity
                obj = cp.Minimize(
                    cp.sum_squares(cp.sum(w) - self.underdetermined_opt)
                )
            elif is_listlike(self.underdetermined_opt):
                wtarget = asarray(self.underdetermined_opt)
                obj = cp.Minimize(
                    cp.sum_squares(wtarget - w)
                )
            elif not is_string(self.underdetermined_opt):
                raise NameError(
                    f"`{self.underdetermined_opt}` is not a underdetermined_opt option."
                )

            elif self.underdetermined_opt == 'max':
                obj = cp.Maximize(cp.sum(w))
            elif self.underdetermined_opt == 'min':
                obj = cp.Minimize(cp.sum(w))
            elif self.underdetermined_opt == 'var':
                obj = cp.Minimize(w - cp.sum(w)/w.size)

            else:
                raise NameError(
                    f"`{self.underdetermined_opt}` is not a underdetermined_opt option."
                )
            prob = cp.Problem(obj, constraints)
            cost = prob.solve()
            assert prob.status == cp.OPTIMAL, 'BUG: optimality not achieved.'
            return OptimizeResult(
                x=w.value,
                cost=cost,
                fun=np.zeros(capture_x.size),
                jac=np.zeros((capture_x.size, self.bg_ints_.size)),
                grad=np.zeros(capture_x.size),
                optimality=0.0,
                active_mask=0,
                nfev=1,
                njev=None,
                status=5,
                message='Optimality achieved and used `underdetermined_opt`',
                success=True
            )

    def _init_sample(self, capture_x, bounds, return_result=False):
        # weighted fit is better for substitution types of fits
        if self.weighted_init_fit_:
            result = lsq_linear(
                self.fit_weights_[:, None] * self.A_,
                self.fit_weights_ * (capture_x - self.q_bg_),
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        else:
            result = lsq_linear(
                self.A_, (capture_x - self.q_bg_),
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        # return fitted intensity (w)
        if return_result:
            return result
        return result.x

    def _objective(self, w, excite_x):
        x_pred = self.get_excitation(w)
        return self.fit_weights_ * (excite_x - x_pred)

    def _process_X(self, X):
        """
        Returns photon capture and excitation value.
        """
        X = optional_to(X, ureg(None).units)
        X = check_array(X)
        # check that input shape is correct
        if X.shape[1] != self.photoreceptor_model_.n_opsins:
            raise ValueError("Shape of input is different from number"
                             "of photoreceptors.")

        # use inverse of excitation function
        capture_X = self.photoreceptor_model_.inv_excitefunc(X)
        excite_X = X
        return capture_X, excite_X

    @property
    def input_units(self):
        return ureg(None).units

    @property
    def fitted_X_(self):
        return self.fitted_excite_X_

    @property
    def X_(self):
        return self.excite_X_


@inherit_docstrings
class TransformExcitationFit(IndependentExcitationFit):
    """
    Class to fit a linear transformation of
    (relative) photoreceptor excitations for each sample independently.

    Parameters
    ----------
    linear_transform : numpy.ndarray, optional
        A linear transformation of the photoreceptor excitation space
        (n_prs, m_space).
    inv_transform : numpy.ndarray, optional
        The inverse of `linear_transform`. If not given, the inverse
        will be estimated using `np.linalg.inv`.
    photoreceptor_model : dreye.Photoreceptor, optional
        A photoreceptor model that implements the `capture`, `excitation`,
        `excitefunc` and `inv_excitefunc` methods. If None,
        a fake photoreceptor model will be created with three different
        photoreceptor types.
    fit_weights : array-like, optional
        Weighting of the importance of each photoreceptor type in the model.
        If None, weighting will be equal between all photoreceptor types.
        Must be same length as the number of photoreceptor types
    background : dreye.Signal, optional
        The spectral distribution of the background illuminant.
    measured_spectra : dreye.MeasuredSpectraContainer, optional
        Container with all available LEDs and their measured spectra. If
        None, a fake LED measurement will be created with intensities
        ranging from 0 to 100 microphotonflux.
    max_iter : int, optional
        The number of maximum iterations. This is passed directly to
        `scipy.optimize.lsq_linear` and `scipy.optimize.least_squares`.
    bg_ints : numpy.ndarray, optional
        The intensity values for each LED, when the relative capture of each
        photoreceptor equals one (i.e. background intensity).
        This will prevent fitting of the
        LED intensities if the background LED intensities
        are preset and the relative capture is 1.
    fit_only_uniques : bool, optional
        If True, use `numpy.unique` to select only the unique samples
        for fitting before transforming X back to the full array.
    ignore_bounds : bool, optional
        If True, ignore the bounds of the LED intensities. Howerver, a zero
        LED intensity bound will always exist.
    lsq_kwargs : dict, optional
        Keyword arguments passed directly to `scipy.optimize.least_squares`.

    Attributes
    ----------
    photoreceptor_model_ : dreye.Photoreceptor
        The photoreceptor model used for fitting. This will be the same if
        as `photoreceptor_model` if it is a `dreye.Photoreceptor` instance.
    measured_spectra_ : dreye.MeasuredSpectraContainer (n_leds)
        Measured spectrum container used for fitting. This will be the same
        if as `measured_spectra` if a `dreye.MeasuredSpectraContainer` instance
        was passed.
    bounds_ : numpy.ndarray (n_leds)
        The LED intensity bounds used for fitting.
    background_ : dreye.Spectrum
        The background used for calculating the relative photon capture.
    A_ : numpy.ndarray (n_prs, n_leds)
        The relative photon capture of each normalized LED spectrum.
    capture_X_ : numpy.ndarray (n_samples, n_prs)
        The current relative photon capture values used for fitting.
    excite_X_ : numpy.ndarray (n_samples, n_prs)
        The current photoreceptor excitation values used for fitting.
    transform_X_ : numpy.ndarray (n_samples, m_space)
        The current linear transformation of the photoreceptor excitations
        used for fitting.
    fitted_intensities_ : numpy.ndarray
        Intensities fit in units of `measured_spectra_.intensities.units`
    fitted_capture_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated relative photon capture values after fitting.
    fitted_excite_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated photoreceptor excitations after fitting.
    fitted_transform_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated linear transform of the photoreceptor excitations
        after fitting.
    """

    _deprecated_kws = {
        **IndependentExcitationFit._deprecated_kws,
        "fit_to_transform": None
    }
    _fit_to_transform = True

    # same length as X but not X or fitted X
    _X_length = IndependentExcitationFit._X_length + [
        'excite_X_',
        'fitted_excite_X_'
    ]

    def __init__(
        self,
        *,
        linear_transform=None,  # array
        inv_transform=None,  # array
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        unidirectional=False,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        lsq_kwargs=None,
        background_external=None, 
        intensity_bounds=None, 
        wavelengths=None, 
        capture_noise_level=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            max_iter=max_iter,
            unidirectional=unidirectional,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            background_external=background_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
        self.linear_transform = linear_transform
        self.inv_transform = inv_transform

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.transform_X_ = X
        if self.linear_transform is None:
            self.W_ = np.eye(X.shape[1])
            self.inv_transform = self.W_
        else:
            self.W_ = asarray(self.linear_transform)
            assert self.W_.shape[0] == X.shape[1], (
                "Linear transform shape does not match X"
            )

        if self.inv_transform is None:
            self.Winv_ = np.linalg.inv(self.W_)
        else:
            self.Winv_ = asarray(self.inv_transform)
            assert self.Winv_.shape[1] == X.shape[1], (
                "Inverse transform shape does not match X"
            )
            assert self.Winv_.shape[0] == self.W_.shape[1], (
                "Inverse matrix row size must match matrix columns."
            )

        super().fit(X @ self.Winv_)
        # overwrite current X
        self.fitted_transform_X_ = self.fitted_excite_X_ @ self.W_
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return excite_X @ self.Winv_

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    @property
    def X_(self):
        return self.transform_X_

    def _objective(self, w, excite_x):
        x_pred = self.get_excitation(w)
        excite_x = excite_x @ self.W_
        x_pred = x_pred @ self.W_
        return self.fit_weights_ * (excite_x - x_pred)


@inherit_docstrings
class NonlinearTransformExcitationFit(IndependentExcitationFit):
    """
    Class to fit a nonlinear transformation of
    (relative) photoreceptor excitations for each sample independently.

    Photoreceptor model and measured_spectra must produce dimensionless
    captures.
    """

    _deprecated_kws = {
        **IndependentExcitationFit._deprecated_kws,
        "fit_to_transform": None
    }
    _fit_to_transform = True

    # same length as X but not X or fitted X
    _X_length = IndependentExcitationFit._X_length + [
        'excite_X_',
        'fitted_excite_X_'
    ]

    def __init__(
        self,
        *,
        transform_func=None,  # array
        inv_func=None,  # array
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        unidirectional=False,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        lsq_kwargs=None,
        background_external=None, 
        intensity_bounds=None,
        wavelengths=None, 
        capture_noise_level=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            max_iter=max_iter,
            unidirectional=unidirectional,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            background_external=background_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
        self.transform_func = transform_func
        self.inv_func = inv_func

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.transform_X_ = X
        if self.transform_func is None:
            def transform_func(X):
                return X
            self.transform_func_ = transform_func
        else:
            assert is_callable(self.transform_func), (
                "`transform_func` must be callable."
            )
            self.transform_func_ = self.transform_func

        if self.inv_func is None and self.transform_func is None:
            self.inv_func_ = transform_func
        elif self.inv_func is None:
            raise DreyeError("Supply `inv_func`, finding "
                             "inverse function not yet implemented.")
        else:
            assert is_callable(self.inv_func), (
                "`inv_func` must be callable."
            )
            self.inv_func_ = self.inv_func

        super().fit(self.inv_func_(X))
        # overwrite current X
        self.fitted_transform_X_ = self.transform_func_(self.fitted_excite_X_)
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return self.inv_func_(excite_X)

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    @property
    def X_(self):
        return self.transform_X_

    def _objective(self, w, excite_x):
        x_pred = self.get_excitation(w)
        excite_x = self.transform_func_(excite_x)
        x_pred = self.transform_func_(x_pred)
        return self.fit_weights_ * (excite_x - x_pred)