"""
Class to calculate various metrics given
a photoreceptor model and measured spectra
"""

import numpy as np

from dreye.utilities.abstract import _InitDict, inherit_docstrings
from dreye.estimators.base import _PrModelMixin
from dreye.estimators.utils import get_optimal_capture_samples, get_source_idcs, get_source_idx, get_spanning_intensities
from dreye.estimators.metric_functions import compute_est_score, compute_metric_for_samples, compute_peak_set, get_metrics
from dreye.estimators.silent_substitution import BestSubstitutionFit
from dreye.estimators.excitation_models import IndependentExcitationFit


# TODO add old metrics back in new format


@inherit_docstrings
class Metrics(_InitDict, _PrModelMixin):
    """
    Metrics to compute the "goodness-of-fit" for various light source
    combinations.
    """
    # never ignore bounds
    ignore_bounds = False

    def __init__(
        self, 
        combos,
        photoreceptor_model,
        measured_spectra,
        *,
        intensity_bounds=None,
        background=None,
        wavelengths=None, 
        capture_noise_level=None
    ):
        # set init
        self.combos = combos
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.intensity_bound = intensity_bounds
        self.wavelengths = wavelengths
        self.background = background
        self.capture_noise_level = capture_noise_level

        self._set_objs()

    def reset_background(self, background):
        self.background = background
        self._set_objs(do_combos=False, do_samples=False)
        return self

    def reset_combos(self, combos):
        self.combos = combos
        self._set_objs(do_pr_model_objs=False, do_optimal=False, do_samples=False)
        return self

    def reset_intensity_bounds(self, intensity_bounds):
        self.intensity_bounds = intensity_bounds
        self._set_objs(do_combos=False, do_optimal=False)

    def _set_objs(
        self, do_pr_model_objs=True, do_combos=True, 
        do_optimal=True, do_samples=True
    ):
        # compute pr model objects
        if do_pr_model_objs:
            self._set_pr_model_related_objects()
        # compute source idcs
        if do_combos:
            self.source_idcs_ = get_source_idcs(
                self.measured_spectra_.names, self.combos
            )
        # compute perfect system
        if do_optimal:
            self.optimal_q_samples_ = get_optimal_capture_samples(
                self.photoreceptor_model_, 
                self.background_
            )
        # compute sample excitations/captures
        if do_samples:
            self.intensity_samples_ = get_spanning_intensities(
                self.intensity_bounds_
            )

    def capture_for_selected_combo(self, source_idx):  # sample func
        """
        Get capture values given selected LED set.
        """
        source_idx = get_source_idx(
            self.measured_spectra_.names, source_idx, 
            asbool=True
        )
        w = self.intensity_samples_.copy()
        w[:, ~source_idx] = 0
        return self.get_capture(w.T)

    def excitation_for_selected_combo(self, source_idx):  # sample func
        """
        Get excitation values given selected LED set.
        """
        q = self.capture_for_selected_combo(source_idx)
        return self.photoreceptor_model_.excitefunc(q)

    def compute_gamuts(self, at_overall_q=None, relative=True, seed=None):
        """
        Compute Gamuts
        """
        if relative:
            relative_to = self.optimal_q_samples_
        else:
            relative_to = None
        
        return get_metrics(
            self.source_idcs_, 
            self.measured_spectra_.names, 
            None, 
            compute_metric_for_samples, 
            'gamut', 
            metric='gamut',
            sample_func=self.capture_for_selected_combo,
            gamut_metric='mean_width', 
            at_overall_q=at_overall_q, 
            relative_to=relative_to,
            center=True,
            centered=True,
            seed=seed,
        )

    def compute_best_substitutions(self, **kwargs):
        """
        Maximum contrast achievable for different combinations of measured spectra.
        """
        X = np.eye(self.photoreceptor_model_.n_opsins).astype(bool)
        return compute_est_score(
            self.source_idcs_, 
            X, 
            self.measured_spectra_, 
            self.photoreceptor_model_, 
            self.background_, 
            BestSubstitutionFit, 
            cols=self.photoreceptor_model_.labels, 
            score_method='feature_scores', 
            name='best_substitution', 
            normalize=False,
            score_kws={'method':'rel', 'aggfunc':'max'},
            intensity_bounds=self.intensity_bounds_, 
            **kwargs
        )

    def compute_excitation_scores(self, X, method='r2', **kwargs):
        """
        Independent excitation fits for different measured spectra combos. 

        See Also
        --------
        IndependentExcitationFit
        """
        return compute_est_score(
            self.source_idcs_, 
            X, 
            self.measured_spectra_, 
            self.photoreceptor_model_, 
            self.background_, 
            IndependentExcitationFit, 
            cols=self.photoreceptor_model_.labels, 
            score_method='feature_scores', 
            name='independent_excitation', 
            normalize=False,
            score_kws=dict(method=method),  # just r2 scores for each photoreceptor
            intensity_bounds=self.intensity_bounds_, 
            **kwargs
        )

    def compute_excitation_scores_for_spectra(
        self, 
        spectra, 
        method='r2',
        **kwargs
    ):
        """
        Independent excitation fits for different measured spectra combos
        given a set of spectra.

        See Also
        --------
        IndependentExcitationFit
        """
        X = self.photoreceptor_model_.excitation(
            spectra, 
            background=self.background_, 
            return_units=False
        )
        return self.compute_excitation_scores(X, method=method, **kwargs)


    def compute_best_peak_difference(
        self, 
        as_string=False,
        as_idx=False,
    ):
        """
        Compute peaks of measured spectra that best match peaks of 
        opsin sensitivities.
        """
        return compute_peak_set(  # formerly compute_as_peaks
            self.photoreceptor_model_.sensitivity, 
            self.measured_spectra_.normalized_spectra,
            as_string=as_string, as_idx=as_idx
        )

    # compute peaks - given available combos - or not?
    # compute for estimators - separate