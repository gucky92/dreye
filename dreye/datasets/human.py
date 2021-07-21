"""
Load human spectral sensitivity
"""

import os
import pandas as pd
from dreye import DREYE_DIR, Signals
from dreye.core.photoreceptor import create_photoreceptor_model

D2_FILENAME = os.path.join(DREYE_DIR, 'datasets', 'linss2_10e_1.csv')
D10_FILENAME = os.path.join(DREYE_DIR, 'datasets', 'linss10e_1.csv')


def load_dataset(as_pr_model=False, d10=False, **kwargs):
    """
    Load human spectral sensitivity.

    Parameters
    ----------
    as_pr_model : bool, optional
        Whether to return a `dreye.Photoreceptor`. If False,
        returns a long-format `pandas.DataFrame`. Defaults to False.
    d10 : bool, optional
        Whether to return the 10 degree measurements. Otherwise, fetches the
        2 degree measurements.
    kwargs : dict
        Keyword arguments passed to `dreye.create_photoreceptor_model`.

    Returns
    -------
    df : `pandas.DataFrame` or `dreye.Photoreceptor`
        A long-format `pandas.DataFrame` with the following columns:
            * `wavelengths`
            * `opsin`
            * `value`
        Or a `dreye.Photoreceptor` instance.

    References
    ----------
    .. [1] Stockman, A, MacLeod, D.I.A, Johnson N.E. (1993)
        Spectral sensitivities of the human cones.
        J. Opt. Soc. Am. A 10, 12, 2491-2521.
    """
    if d10:
        data = pd.read_csv(D10_FILENAME, header=None)
    else:
        data = pd.read_csv(D2_FILENAME, header=None)

    data.columns = ['wavelengths', 'L', 'M', 'S']
    if as_pr_model:
        data = data.set_index('wavelengths')
        data = Signals(data.fillna(0), domain_units='nm')
        return create_photoreceptor_model(data, name='human', **kwargs)

    return data.melt(['wavelengths'], var_name='opsin').fillna(0)
