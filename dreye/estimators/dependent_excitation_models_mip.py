"""
Dependent Excitation Models
"""

import warnings
import numpy as np
from scipy.optimize import least_squares
from mip import *

from dreye.utilities.abstract import inherit_docstrings
from dreye.estimators.excitation_models import IndependentExcitationFit


@inherit_docstrings
class DependentExcitationFit(IndependentExcitationFit):

    def __init__(
        self,
        *,
        independent_layers=None,  # int
        layer_assignments=None,  # list of lists or array-like
        bit_depth=1,
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
        seed=None
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
            wavelengths=wavelengths
        )
        self.independent_layers = independent_layers
        self.layer_assignments = layer_assignments
        self.bit_depth = bit_depth
        self.seed = seed

    def _fit(self, X):
        if self.independent_layers is None and self.layer_assignments is None:
            self._independent_layers_ = len(self.measured_spectra_)
            self._layer_assignments_ = [[idx] for idx in range(len(self.measured_spectra_))]
        elif self.independent_layers is None:
            self._independent_layers_ = len(self.layer_assignments)
            self._layer_assignments_ = self.layer_assignments
            # TODO assert right type and indices exist in measured_spectra
        elif self.layer_assignments is None:
            self._independent_layers_ = self.independent_layers
            self._layer_assignments_ = [
                list(range(len(self.measured_spectra_))) 
                for _ in range(self._independent_layers_)
            ]
        else:
            assert len(self.layer_assignments) == self.independent_layers
            self._independent_layers_ = self.independent_layers
            self._layer_assignments_ = self.layer_assignments
            # TODO assert right type and indices exist in measured_spectra

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)

        # if only fit uniques used different iterator
        if self.fit_only_uniques:
            # get uniques
            _, xidcs, xinverse = np.unique(
                self.capture_X_, axis=0, return_index=True, return_inverse=True
            )
            fitted_intensities, layer_intensities, pixel_strength = self._fit_sample(
                self.capture_X_[xidcs], self.excite_X_[xidcs]
            )
            fitted_intensities = fitted_intensities[xinverse]
            layer_intensities = layer_intensities[xinverse]
        else:
            fitted_intensities, layer_intensities, pixel_strength = self._fit_sample(
                self.capture_X_, self.excite_X_
            )

        # len(measured_spectra) x independent_layers, len(X) x independent_layers
        self.layer_intensities_, self.pixel_strength_, self.fitted_intensities_ = (
            layer_intensities, pixel_strength, fitted_intensities
        )
        self.fitted_excite_X_ = self.get_excitation(self.fitted_intensities_.T)
        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )

        return self

    def _reformat_intensities(self, w, **kwargs):
        # len(measured_spectra) x independent_layers, len(X) x independent_layers
        ws, pixel_strength = self._format_intensities(w, **kwargs)
        # len(X) x len(measured_spectra)
        return pixel_strength @ ws.T
        # return (ws[None, ...] * pixel_strength[:, None, ...]).sum(axis=-1)

    def _format_intensities(self, w, ws=None, pixel_strength=None):
        offset = 0
        if ws is None:
            ws = np.zeros((len(self.measured_spectra_), self._independent_layers_))
            for idx, source_idcs in enumerate(self._layer_assignments_):
                ws[source_idcs, idx] = w[offset:offset+len(source_idcs)]
                offset += len(source_idcs)
        
        if pixel_strength is None:
            pixel_strength = w[offset:].reshape(-1, self._independent_layers_)
            # TODO Find better method to handle this (MIP here)
            pixel_strength = np.round(pixel_strength * self.bit_depth, 0) / self.bit_depth
        return ws, pixel_strength

    def _fit_sample(self, capture_x, excite_x):
        np.random.seed(self.seed)
        # adjust bounds if necessary
        bounds = list(self.intensity_bounds_)
        if self._unidirectional_:
            if np.all(capture_x >= self.capture_border_):
                bounds[0] = self.bg_ints_
            elif np.all(capture_x <= self.capture_border_):
                bounds[1] = self.bg_ints_
        # find initial w0 using linear least squares by using the mean capture across all pixels
        w0 = self._init_sample(capture_x.mean(0), bounds)
        # add independent layer dimensions
        w0s = []
        bounds0 = []
        bounds1 = []
        for source_idcs in self._layer_assignments_:
            w0s.append(w0[source_idcs])
            bounds0.append(bounds[0][source_idcs])
            bounds1.append(bounds[1][source_idcs])

        # pixel strength values
        n_pixels = len(capture_x)
        bounds0.append([0]*n_pixels*self._independent_layers_)
        bounds1.append([1]*n_pixels*self._independent_layers_)
        # random initial values for pixel intensities
        w0s.append(np.random.random(n_pixels*self._independent_layers_))

        # TODO make more like EM-algorithm
        # TODO handle pixel bit depth better

        # reformatted w0 and bounds
        w0 = np.concatenate(w0s)
        bounds = (
            np.concatenate(bounds0), 
            np.concatenate(bounds1)
        )
        # fitted result
        result = least_squares(
            self._objective,
            x0=w0,
            args=(excite_x,),
            bounds=bounds,
            max_nfev=self.max_iter,
            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
        )

        if not result.success:
            warnings.warn("Convergence was not accomplished "
                          "for X; "
                          "increase the number of max iterations.", RuntimeWarning)

        layer_intensities, pixel_strength = self._format_intensities(result.x)
        fitted_intensities = self._reformat_intensities(result.x)
        
        return fitted_intensities, layer_intensities, pixel_strength

    def _objective(self, w, excite_x, **kwargs):
        w = self._reformat_intensities(w, **kwargs).T  # len(measured_spectra) x len(X)
        return super()._objective(w, excite_x).ravel()