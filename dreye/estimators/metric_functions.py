"""
various metric functions
"""

import tqdm
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.feature_selection import mutual_info_regression

from dreye.utilities.common import is_numeric
from dreye.utilities import (
    compute_jensen_shannon_similarity, 
    compute_mean_width, is_dictlike
)
from dreye.utilities.barycentric import (
    barycentric_dim_reduction, simplex_plane_points_in_hull
)
from dreye.estimators.utils import get_source_idx, get_source_idx_string


def transform_points(points, B=None):
    """
    Transform `points` with B

    Parameters
    ----------
    points : numpy.ndarray
        2D matrix.
    B : callable or numpy.ndarray, optional
        If callable, `B` is a function: B(points). If B is a
        `numpy.ndarray`, then `B` is treated as a matrix: points @ B.
    """
    if B is None:
        return points
    elif callable(B):
        return B(points)
    else:
        return points @ B


def compute_volume(points):
    """
    Compute the volume from a set of points.
    """
    if (points.ndim == 1) or (points.shape[1] < 2):
        return np.max(points) - np.min(points)
    if points.shape[0] <= points.shape[1]:
        return 0
    try:
        convex_hull = ConvexHull(
            points, qhull_options="QJ Pp"
        )
        return convex_hull.volume
    except Exception:
        points = points - np.mean(points, axis=0)
        convex_hull = ConvexHull(
            points,
            qhull_options="QJ Pp"
        )
        return convex_hull.volume


def compute_continuity(points, bins=100, **kwargs):
    """
    Compute continuity of data by binning. Useful for circular datapoints.

    See Also
    --------
    compute_jss_uniformity
    """
    if (points.ndim == 1) or (points.shape[1] < 2):
        H = np.histogram(points, bins, **kwargs)[0].astype(bool)
    else:
        H = np.histogramdd(points, bins, **kwargs)[0].astype(bool)
    return H.sum() / H.size


def compute_jss_uniformity(points, bins=100, **kwargs):
    """
    Compute how similar the dataset is to a uniform distribution.
    """
    if (points.ndim == 1) or (points.shape[1] < 2):
        H = np.histogram(points, bins, **kwargs)[0]
    else:
        H = np.histogramdd(points, bins, **kwargs)[0]
    H_uniform = np.ones(H.shape)
    return compute_jensen_shannon_similarity(H, H_uniform)


def compute_width(points, n=1000, **kwargs):
    """
    Compute mean width.
    """
    if (points.ndim == 1) or (points.shape[1] < 2):
        return np.max(points) - np.min(points)
    return compute_mean_width(points, n, **kwargs)


def compute_mean_correlation(points):
    # compute correlation of each feature
    cc = np.abs(np.corrcoef(points, rowvar=False))
    return (cc - np.eye(cc.shape[0])).mean()


def compute_mean_mutual_info(points, **kwargs):
    mis = []
    for idx in range(points.shape[1] - 1):
        mi = mutual_info_regression(points[idx], points[idx + 1:], **kwargs)
        mis.append(mi)
    return np.concatenate(mis).mean()


def compute_gamut(
    points,
    gamut_metric='mean_width', 
    at_overall_q=None, 
    relative_to=None,
    center=False,
    **kwargs
):
    """
    Compute absolute gamut
    """
    gamut_metric = get_metric_func(gamut_metric)

    overall_q = points.sum(axis=-1)
    
    if at_overall_q is not None:
        if np.all(overall_q <= at_overall_q):
            return 0
        elif np.all(overall_q > at_overall_q):
            return 0
        # just keep current points if zero q the only one below at_overall_q
        elif np.all(overall_q[overall_q < at_overall_q] == 0):
            pass
        else:
            points = simplex_plane_points_in_hull(points, at_overall_q)
            overall_q = points.sum(axis=-1)
    
    points = points[overall_q != 0]
    points = barycentric_dim_reduction(points)
    if center:
        points -= barycentric_dim_reduction(np.ones((1, points.shape[1]+1)))
    num = gamut_metric(points, **kwargs)
    
    if relative_to is not None:
        denom = compute_gamut(
            relative_to, 
            gamut_metric=gamut_metric, 
            at_overall_q=None,
            relative_to=None,
            center=center,
            **kwargs
        )
        return num / denom
    return num


def compute_peak_metric(
    sensitivity,
    normalized_spectra,
):
    """
    Find peaks of spectra that best match the opsin sensitivities, 
    and calculate the distance between the light source peaks and
    opsin sensitivity peaks.
    """
    pr_max = sensitivity.dmax
    s_max = normalized_spectra.dmax
    diff = np.abs(s_max[:, None] - pr_max[None])
    amin = np.min(diff, axis=0)
    return amin


def compute_peak_set(  # formerly compute_as_peaks
    sensitivity, 
    normalized_spectra,
    as_string=False, as_idx=False
):
    """
    Compute best set according to peaks of opsins and LEDs
    """
    pr_max = sensitivity.dmax
    s_max = normalized_spectra.dmax
    argmin = np.argmin(np.abs(s_max[:, None] - pr_max[None]), axis=0)
    if np.unique(argmin).size != argmin.size:
        argmin_ = argmin
        argmin = []
        for odx, amin in enumerate(argmin_):
            if amin not in argmin:
                argmin.append(amin)
                continue

            argsort = np.argsort(np.abs(pr_max[odx] - s_max))
            for amin_ in argsort:
                if amin_ not in argmin:
                    amin = amin_
                    break
            else:
                warnings.warn("Couldn't find unique set of LEDs for opsin set.")
            argmin.append(amin)
    if as_idx:
        return np.array(argmin)
    
    best = normalized_spectra.labels[argmin]
    if as_string:
        return '+'.join(best)
    return best


def get_metric_func(metric):
    if callable(metric):
        return metric
    elif metric in {'volume', 'vol'}:
        return compute_volume
    elif metric in {'jss_uniformity', 'uniformity_similarity'}:
        return compute_jss_uniformity
    elif metric in {'mean_width', 'mw'}:
        return compute_width
    elif metric in {'continuity', 'cont'}:
        return compute_continuity
    elif metric in {'corr', 'correlation'}:
        return compute_mean_correlation
    elif metric in {'mi', 'mutual_info'}:
        return compute_mean_mutual_info
    elif metric in {'gamut', }:
        return compute_gamut

    raise NameError(
        f"Did not recognize metric `{metric}`. "
        "`metric` must be a callable or an accepted str: "
        "{"
        "'volume', 'vol', 'jss_uniformity', 'uniformity_similarity', "
        "'mean_width', 'mw', 'continuity', 'cont', 'corr', 'correlation', "
        "'mi', 'mutual_info', 'gamut'."
        "}."
    )


def get_source_df(
    source_idcs,  # boolean representation
    source_names, 
):
    source_df = pd.DataFrame(
        source_idcs,
        columns=source_names
    )
    for idx, source_idx in enumerate(source_idcs):
        source_df.loc[idx, 'light_combos'] = get_source_idx_string(source_names, source_idx)
        source_df.loc[idx, 'k'] = np.sum(source_idx)

    source_df.loc[:, 'k'] = source_df['k'].astype(int)
    return source_df


def metric_constructor_helper(
    source_df, 
    name, 
    metric_func,   # accepts light_combos, args and kwargs
    *args, 
    normalize=False,
    cols=None,
    **kwargs
):
    metrics = source_df.copy()
    for idx, row in tqdm.tqdm(source_df.iterrows(), total=len(source_df)):
        metric = metric_func(
            row['light_combos'], 
            *args, **kwargs
        )
        if (cols is None) or is_numeric(metric):
            metrics.loc[idx, 'metric'] = metric
        else:
            metrics.loc[idx, cols] = metric
            metrics.loc[idx, 'metric'] = metric.mean()

    metrics['metric_name'] = name

    if normalize:
        metrics['metric'] /= metrics['metric'].abs().max()
        if cols is not None and not (set(cols) - set(metrics.cols)):
            metrics[cols] /= metrics[cols].abs().max(axis=0)

    return metrics



def get_metrics(
    source_idcs, source_names, cols,
    metric_func, name, B=None,
    normalize=False, B_name=None, **kwargs
):

    source_df = get_source_df(source_idcs, source_names)

    def helper(B):
        return metric_constructor_helper(
            source_df, 
            name, 
            metric_func, 
            B=B,
            normalize=normalize,
            cols=cols,
            **kwargs
        )

    if is_dictlike(B):
        metrics = pd.DataFrame()
        for transformation, B_ in B.items():
            metrics_ = helper(B_)
            metrics_['transformation'] = transformation
            metrics = metrics.append(metrics_, ignore_index=True)
        return metrics
    else:
        metrics = helper(B)
        metrics['transformation'] = B_name
        return metrics



def compute_metric_for_samples(  
    # formerly get_capture_metric
    source_idx,
    sample_func,
    metric='volume', 
    B=None, **kwargs
):
    """
    Compute metric for particular combination of source lights.
    """
    metric_func = get_metric_func(metric)
    # get values and transform
    points = sample_func(source_idx)
    points = transform_points(points, B)
    return metric_func(points, **kwargs)


def compute_constant_metric(
    source_idx, 
    kwargs_func,
    metric_func,
    **kwargs
):
    kws = kwargs_func(source_idx)
    return metric_func(**kws, **kwargs)


def compute_est_score(
    source_idcs, 
    X,
    measured_spectra, 
    photoreceptor_model,
    background, 
    est_cls, 
    cols=None,
    score_method='score',
    name=None,
    normalize=False,
    score_kws={},
    **kwargs
):
    """
    Compute score of estimator
    """

    if name is None:
        name = score_method

    def metric_func(source_idx, B=None, **kwargs):
        assert B is None, "Why is B not None?"
        _source_idx = get_source_idx(measured_spectra.names, source_idx)
        intensity_bounds = kwargs.get('intensity_bounds', None)
        if intensity_bounds is not None:
            intensity_bounds = (
                intensity_bounds[0][_source_idx], 
                intensity_bounds[1][_source_idx]
            )
            kwargs['intensity_bounds'] = intensity_bounds
        est = est_cls(
            photoreceptor_model=photoreceptor_model, 
            measured_spectra=measured_spectra[_source_idx], 
            background=background,
            **kwargs
        )
        est.fit(X)
        return getattr(est, score_method)(**score_kws)

    return get_metrics(
        source_idcs, measured_spectra.names, cols,
        metric_func, name, B=None,
        normalize=normalize, B_name=None, 
        **kwargs
    )