"""various mixin classes
"""

from itertools import product
import numpy as np
import pandas as pd

from dreye.utilities import is_numeric


class SetBaselineMixin:

    # --- misc helper function --- #
    @staticmethod
    def _set_baseline_values(baseline_values, channel_names):
        """function for setting baseline values attribute to
        numpy array. Method is used in _set_values method.
        """

        # check if baseline_values are correct size
        if is_numeric(baseline_values):
            baseline_values = np.array(
                [baseline_values] * len(channel_names)
            )
        else:
            baseline_values = np.array(baseline_values)
            assert len(channel_names) == len(baseline_values)

        return baseline_values.astype(float)


class SetStepMixin(SetBaselineMixin):

    def _set_values(self, values, baseline_values, separate_channels):
        """function to set values attribute to pandas DataFrame.
        """

        # convert values into pandas dataframe object
        if is_numeric(values):
            channel_names = np.array([0])
            baseline_values = self._set_baseline_values(
                baseline_values, channel_names)
            values = np.array([values]).astype(float)

        elif isinstance(values, dict):
            channel_names = np.array(list(values.keys()))
            baseline_values = self._set_baseline_values(
                baseline_values, channel_names)
            if separate_channels:
                _values = []
                for index, ele in enumerate(values.values()):
                    __values = np.array(
                        [baseline_values]*len(ele)
                    ).astype(float)
                    __values[:, index] = np.array(ele).astype(float)
                    _values.extend(__values.tolist())
                values = _values
            else:
                values = list(product(*values.values()))

        elif not isinstance(values, pd.DataFrame):
            values = np.array(values)
            if values.ndim > 1:
                channel_names = np.arange(values.shape[1])
            else:
                channel_names = np.array([0])
            baseline_values = self._set_baseline_values(
                baseline_values, channel_names)

        else:
            channel_names = values.columns
            baseline_values = self._set_baseline_values(
                baseline_values, channel_names)

        return pd.DataFrame(
            np.array(values).astype(float), columns=channel_names
        ), baseline_values


class SetRandomStepMixin(SetBaselineMixin):

    def _set_values(self, values, baseline_values, values_probs):
        """return values in dictionary format and baseline_values as
        numpy.array
        """

        if isinstance(values, dict):
            if values_probs is None:
                values_probs = {}
            else:
                assert isinstance(values_probs, dict), \
                    'values probs must be dict'

            channel_names = np.array(list(values.values()))
            # transform each element into numpy array
            for key, ele in values.items():
                # convert element to numpy array
                ele = np.array(ele)
                values[key] = ele

                # check values probs array
                if values_probs.get(key, None) is None:
                    values_probs[key] = np.ones(len(ele))/len(ele)
                else:
                    values_prob = np.array(values_probs[key])
                    values_prob = values_prob / np.sum(values_prob)
                    assert len(values_prob) == len(ele), \
                        'values prob not same length as values.'
                    values_probs[key] = values_prob

        else:
            channel_names = np.array([0])

            values = np.array(values)
            assert values.ndim == 1, \
                'if array-like values must be one-dimensional'

            if values_probs is None:
                values_probs = np.ones(len(values))/len(values)
            else:
                values_probs = np.array(values_probs) / np.sum(values_probs)
                assert len(values_probs) == len(values), \
                    'values prob not same length as values.'

            values = {channel_names[0]: values}
            values_probs = {channel_names[0]: values_probs}

        baseline_values = self._set_baseline_values(
            baseline_values, channel_names)

        return values, baseline_values, values_probs