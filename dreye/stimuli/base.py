"""Base class for stimuli
"""

from abc import ABC, abstractmethod
import random
import warnings
import inspect

import numpy as np
import pandas as pd

from dreye.io import read_json, write_json
from dreye.utilities import is_numeric, is_listlike, asarray, is_callable
from dreye.utilities.abstract import inherit_docstrings
from dreye.err import DreyeError
from dreye.stimuli.variables import (
    DUR_KEY, DELAY_KEY, PAUSE_KEY, CHAINED_STIM_INDEX
)
from dreye.stimuli.plotting import StimPlottingMixin


def _check_events(df):
    """
    Check that event dataframe contains columns DELAY_KEY and DUR_KEY
    """

    if df is None:
        df = pd.DataFrame()
    elif not isinstance(df, pd.DataFrame):
        raise DreyeError('Events frame must be dataframe.')

    elif len(df) == 0:

        cols = list(df.columns)
        if (DELAY_KEY in cols and DUR_KEY in cols and PAUSE_KEY in cols):
            pass
        else:
            if DELAY_KEY not in cols:
                cols.append(DELAY_KEY)
            if DUR_KEY not in cols:
                cols.append(DUR_KEY)
            if PAUSE_KEY not in cols:
                cols.append(PAUSE_KEY)
            df = pd.DataFrame(columns=cols)

    else:
        assert not ({DELAY_KEY, DUR_KEY} - set(df.columns)), \
            'event frame must contain column start and dur.'
        if PAUSE_KEY not in df.columns:
            df[PAUSE_KEY] = 0.0

        # stringify columns - and copy dataframe WHY?
        cols = {
            col: str(col) for col in df.columns
            if not isinstance(col, str)
        }
        if cols:
            df = df.rename(columns=cols)

    return df


@inherit_docstrings
class BaseStimulus(ABC, StimPlottingMixin):
    """
    Abstract base stimulus class.

    For subclassing, you only need to implement a `create` method
    that assigns the following attributes:

    * `signal` (`numpy.ndarray`)
    * `events` (`pandas.DataFrame`)
    * `metadata` (`dict`)

    Parameters
    ----------
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    rate : numeric
        The refresh rate.
    seed : int
        A random seed.
    kwargs : dict
        Keyword arguments that are kept as attributes. These can be seen
        as specific settings for particular stimuli.
    """

    time_axis = 0
    channel_axis = -1
    _add_mean_to_events = True

    def __init__(
        self, *, estimator=None,
        rate=None, seed=None, subsample=None,
        **kwargs
    ):
        self.estimator = estimator
        self.subsample = subsample
        self.seed = seed
        self.rate = rate
        if rate is not None:
            assert is_numeric(rate), 'rate must be numeric'
        if seed is not None:
            assert is_numeric(seed), 'seed must be numeric'
        if estimator is not None:
            assert hasattr(estimator, 'fit_transform'), (
                'Estimator requires methods `fit` and `transform`.'
            )

        # directly set elements
        for key, ele in kwargs.items():
            try:
                setattr(self, key, ele)
            except AttributeError:
                raise DreyeError(f"key `{key}` is reserved; "
                                 "change argument name.")

        self._stimulus = None
        self._signal = None
        self._fitted_signal = None
        self._metadata = None
        self._events = None
        self._rng = np.random.default_rng(self.seed)

        # if settings already exists simply update dictionary
        settings = {
            'rate': self.rate,
            'seed': self.seed,
            'estimator': self.estimator,
            'subsample': self.subsample,
            **kwargs
        }
        if hasattr(self, '_settings'):
            self._settings.update(
                settings
            )
        else:
            self._settings = settings

    @property
    def rng(self):
        return self._rng

    @property
    def _plot_attrs(self):
        plot_attrs = super()._plot_attrs
        if hasattr(self.estimator, '_X_length'):
            plot_attrs = (
                plot_attrs[:-1] + self.estimator._X_length + plot_attrs[-1:]
            )
        return plot_attrs

    @abstractmethod
    def create(self):
        """
        Create `signal`, `events`, and `metadata` attributes.
        """

    def transform(self):
        """
        Transform `signal` to `stimulus` using the `estimator` `fit_transform`
        method or simply set them equal if no `estimator` was passed.
        """

        if self.estimator is None:
            self._stimulus = self.signal
        else:
            if hasattr(self.estimator, 'map'):
                self._stimulus = self.estimator.map(self.signal)
            elif self.signal.ndim != 2:
                raise ValueError(
                    'Signal must be two-dimensional, '
                    'but is {self.signal.ndim}.'
                )
            else:
                self._stimulus = self.estimator.fit_transform(self.signal)

        if (
            hasattr(self.estimator, 'fitted_X_')
            and hasattr(self.estimator, 'X_')
        ):
            if self.estimator.X_.shape == self.signal.shape:
                self._fitted_signal = self.estimator.fitted_X_
            else:
                self._fitted_signal = self.signal
        else:
            self._fitted_signal = self.signal

        events = _check_events(self._events)
        # if channel names not in events add signal
        # HERE no overlap at all in names
        if not set(events.columns) & set(self.ch_names):
            events = self._add_to_events(
                events, self.ch_names, self.signal, '',
                'signal'
            )
        # MUST include all channel names
        elif (
            not set(self.ch_names) - set(events.columns)
            and hasattr(self, '_attrs_to_event_labels_mapping')
        ):
            # add signal to event labels mapping
            self._attrs_to_event_labels_mapping['signal'] = self.ch_names
        # add stimulus and fitted_signal to events
        events = self._add_to_events(
            events, self.ch_names, self.fitted_signal, 'fitted_',
            'fitted_signal'
        )
        if hasattr(self.estimator, '_X_length'):
            for attr in self.estimator._X_length:
                # special cases (handles photoreceptor model and spms)
                if (
                    'intensities_' in attr
                    and hasattr(self.estimator, 'measured_spectra_')
                ):
                    labels = self.estimator.measured_spectra_.names
                    prefix = attr.replace('intensities_', '')
                elif (
                    'excite_X_' in attr
                    and hasattr(self.estimator, 'photoreceptor_model_')
                ):
                    labels = self.estimator.photoreceptor_model_.labels
                    prefix = attr.replace('excite_X_', '')
                elif (
                    'capture_X_' in attr
                    and hasattr(self.estimator, 'photoreceptor_model_')
                ):
                    labels = self.estimator.photoreceptor_model_.labels
                    prefix = attr.replace('capture_X_', 'q_')
                else:
                    labels = None
                    prefix = attr
                events = self._add_to_events(
                    events, labels, getattr(self.estimator, attr), prefix,
                    attr
                )

        # ensure check again (stringify etc.)
        events = _check_events(events)
        self._events = events

    # --- short hand for metadata dictionary --- #

    def __getitem__(self, index):
        """
        Get metadata item.
        """

        return self.metadata[index]

    def __setitem__(self, index, value):
        """
        Set metadata item.
        """

        self.metadata[index] = value

    def __getattr__(self, name):
        if 'estimator' in vars(self) and hasattr(self.estimator, name):
            return getattr(self.estimator, name)
        else:
            raise AttributeError(
                f"`{type(self).__name__}` instance has not attribute `{name}`."
            )

    # --- properties of stimulus --- #

    @property
    def name(self):
        """
        Short-hand for the stimulus class name
        """

        return type(self).__name__

    @property
    def metadata(self):
        """
        A dictionary of the metadata
        """

        if self._stimulus is None:
            self.transform()

        if self._metadata is None:
            return {}

        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """
        Set metadata dictionary.

        This can only be done once.
        """

        if self._metadata is not None:
            warnings.warn('`metadata` already set. Cannot reset')
            return

        assert isinstance(value, dict), "`metadata` must be dict."

        self._metadata = value

    @property
    def ch_names(self):
        """
        List of names for each channel.
        """
        if self.channel_axis is None:
            return [0]
        return list(range(self.signal.shape[self.channel_axis]))

    @property
    def events(self):
        """
        A `pandas.DataFrame` of events in the stimulus;
        each row represents a single event
        """

        if self._stimulus is None:
            self.transform()

        return self._events

    @events.setter
    def events(self, value):
        """
        Set events `pandas.DataFrame`.

        This can only be done once.
        """

        if self._events is not None:
            warnings.warn('Events already set. Cannot reset')
            return

        assert isinstance(value, pd.DataFrame)

        self._events = value

    def subsample_events(self, n=None, frac=None, seed=None):
        """
        Subsample `events` attribute to create a new stimulus, with
        shuffled and subsampled events.

        Parameters
        ----------
        n : int, optional
            Number of events to sample. Cannot be used with `frac`.
        frac : float, optional
            Fraction of events to sample. Cannot be used with `n`.
        seed : int
            Seed for random number generator.

        Returns
        -------
        obj : subclass of `BaseStimulus`
        """
        # get data dictionary
        data = self.to_dict()
        # stimulus, signal, fitted_signal reassigned to data
        # get events dataframe and sort by delay key
        events = pd.DataFrame(data.pop('events')).sort_values(DELAY_KEY)
        stimulus = data.pop('stimulus')
        signal = data.pop('signal', stimulus)
        fitted_signal = data.pop('fitted_signal', signal)
        # get number of events to remove
        n = (1 if ((n is None) and (frac is None)) else n)
        if n is None:
            n = int(frac * events.shape[0])
        rm_n = events.shape[0] - n

        # get indices of events to remove
        rm_idcs = events.sample(
            n=rm_n, replace=False, random_state=seed, axis=0
        ).index

        # remove events from traces
        # everything happens inplace
        for rm_idx in rm_idcs:
            rm_row = events.loc[rm_idx]
            rm_delay = rm_row[DELAY_KEY]
            rm_dur = rm_row[[DUR_KEY, PAUSE_KEY]].sum()
            # get idcs to keep
            rm_idx1 = int(rm_delay * self.rate)
            rm_idx2 = int((rm_delay + rm_dur) * self.rate)
            idcs = np.arange(signal.shape[self.time_axis])
            keep_bool = (idcs < rm_idx1) | (idcs >= rm_idx2)
            # remove signal, stimulus, fitted_signal part
            signal = np.compress(
                keep_bool, signal, axis=self.time_axis
            )
            stimulus = np.compress(
                keep_bool, stimulus, axis=self.time_axis
            )
            fitted_signal = np.compress(
                keep_bool, fitted_signal, axis=self.time_axis
            )
            if hasattr(self.estimator, '_X_length'):
                for attr in self.estimator._X_length:
                    arr = getattr(self.estimator, attr)
                    setattr(
                        self.estimator, attr,
                        np.compress(
                            keep_bool, arr, axis=self.time_axis
                        )
                    )
            # remove event
            events.drop(index=rm_idx, inplace=True)
            # change delay in events after removed events
            events.loc[
                events.loc[:, DELAY_KEY] > rm_delay, DELAY_KEY
            ] -= rm_dur

        data['events'] = events
        data['signal'] = signal
        data['stimulus'] = stimulus
        data['fitted_signal'] = fitted_signal
        return type(self).from_dict(data)

    def time2frame(self, key=DELAY_KEY):
        """
        Get frame idcs for delay period of each event.
        """

        if self.rate is None:
            return self.events[key].round(0).astype(int)
        else:
            return (self.events[key] * self.rate).round(0).astype(int)

    @property
    def signal(self):
        """
        Signal prior to transforming to `stimulus`.
        """

        if self._signal is None:
            self.create()

        return self._signal

    @signal.setter
    def signal(self, value):
        """
        Set signal.

        This setter can only be used once.
        """

        if self._signal is not None:
            warnings.warn('Signal already set. Cannot reset')
            return

        assert isinstance(value, np.ndarray)

        self._signal = value

    @property
    def fitted_signal(self):
        """
        Fitted `signal` instance after applying transforming using
        the `estimator`.

        This only differs from `signal` if the `estimator` has
        the property `fitted_X_`.
        """

        if self._fitted_signal is None:
            self.transform()

        return self._fitted_signal

    @property
    def stimulus(self):
        """
        Stimulus array after applying `fit_transform` to `signal`.

        This array is usually suited for sending values directly
        to a `hardware.AbstractSystem`.
        """

        if self._stimulus is None:
            self.transform()

        return self._stimulus

    @property
    def other_shape(self):
        """
        Shape of stimulus (excluding `time_axis`).
        """

        shape = list(self.stimulus.shape)
        shape.pop(self.time_axis)
        return tuple(shape)

    @property
    def time_len(self):
        """
        Length of stimulus along `time_axis`.
        """

        return self.stimulus.shape[self.time_axis]

    @property
    def channel_len(self):
        """
        Length of stimulus along `channel_axis`.
        """
        if self.channel_axis is None:
            return 1
        return self.stimulus.shape[self.channel_axis]

    @property
    def timestamps(self):
        """
        Timestamps array of stimulus.
        """

        length = self.signal.shape[self.time_axis]
        arr = np.arange(0, length)

        if self.rate is None:
            return arr
        else:
            return arr / self.rate

    @property
    def duration(self):
        """
        Duration of stimulus.
        """

        length = self.signal.shape[self.time_axis]

        if self.rate is None:
            return length
        else:
            return length / self.rate

    @property
    def settings(self):
        """
        A dictionary of the settings initialized with.
        """

        if not hasattr(self, '_settings'):
            return {}

        return self._settings

    # --- serialization methods --- #

    def to_dict(self):
        """
        Dictionary format of stimulus class
        """

        dictionary = {
            'stimulus': self.stimulus,
            'metadata': self.metadata,
            'events': self.events.to_dict('list'),
            'settings': self.settings,
            '_attrs_to_event_labels_mapping':
                self._attrs_to_event_labels_mapping
        }
        if self.estimator is not None:
            dictionary['signal'] = self.signal
            if not np.allclose(self.signal, self.fitted_signal):
                dictionary['fitted_signal'] = self.fitted_signal

        return dictionary

    @classmethod
    def from_dict(cls, data):
        """
        Create stimulus class from dictionary.
        """
        # remove settings no longer required
        try:
            self = cls(**data['settings'])
        except TypeError:
            params = inspect.signature(cls.__init__).parameters
            settings = {
                k: v for k, v in data['settings'].items()
                if k in params
            }
            self = cls(**settings)
        self._stimulus = data['stimulus']
        self._signal = data.get('signal', self._stimulus)
        self._fitted_signal = data.get('fitted_signal', self._signal)
        self._metadata = data['metadata']
        self._events = pd.DataFrame(data['events'])
        self._attrs_to_event_labels_mapping = data.get(
            '_attrs_to_event_labels_mapping', {}
        )

        return self

    def save_settings(self, filename):
        """
        Save `settings` dictionary as JSON.
        """

        write_json(filename, self.settings)

    @classmethod
    def load_settings(cls, filename):
        """
        Load JSON file with `settings` dictionary.
        """

        return read_json(filename)

    def save(self, filename):
        """
        Save complete stimulus.
        """

        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        """
        Load complete stimulus.
        """

        return read_json(filename)

    def _add_to_events(self, events, labels, signal, prefix, attr_name):
        """
        Method used to add transformations of signals
        """
        # has to be an array
        signal = asarray(signal)

        if self.channel_axis is None:
            channel_len = 1
        else:
            channel_len = signal.shape[self.channel_axis]

        if labels is None:
            labels = [
                f'{prefix}{idx}'
                for idx in range(channel_len)
            ]
        else:
            labels = [
                f"{prefix}{label}" for label in labels
            ]
        # add opsin channels to events
        # warn if channel names already exist
        truth = (
            set(labels)
            & set(events.columns)
        )
        if truth:
            raise DreyeError('Events Dataframe already contains columns'
                             f' for labels {labels}.')
        # assert sizes match
        assert len(labels) == channel_len
        # if signal high-dimensional then events columns need to be object type
        events_ = {}
        for channel in labels:
            events_[channel] = []

        for idx, row in events.iterrows():
            # get indices for event
            if self.rate is None:
                idcs = asarray([row[DELAY_KEY]]).astype(int)
            else:
                dur_length = row[DUR_KEY] * self.rate
                delay_idx = row[DELAY_KEY] * self.rate
                idcs = (np.arange(dur_length) + delay_idx).astype(int)
            # extract signal from event
            _signal = np.take(signal, idcs, axis=self.time_axis)
            # for each channel in labels take channel in signal
            # and average across time (and channel)
            for jdx, channel in enumerate(labels):
                # taking a list ensures that the axis are still aligned!
                if self.channel_axis is None:
                    if self._add_mean_to_events:
                        value = _signal.mean(
                            axis=(self.time_axis)
                        )
                    else:
                        value = _signal
                else:
                    value = np.take(_signal, [jdx], axis=self.channel_axis)
                    if self._add_mean_to_events:
                        value = value.mean(
                            axis=(self.time_axis, self.channel_axis)
                        )
                    else:
                        value = value.mean(axis=self.channel_axis)
                # assign entry
                events_[channel].append(value)

        events_ = pd.DataFrame(events_, index=events.index)
        # reassign events
        events = pd.concat([events, events_], axis=1)
        # assign attr_name to labels
        if hasattr(self, '_attrs_to_event_labels_mapping'):
            self._attrs_to_event_labels_mapping[attr_name] = labels

        return events


@inherit_docstrings
class DynamicStimulus(BaseStimulus):
    """
    Create a Stimulus dynamically given a create function.

    Parameters
    ----------
    create_func : callable
        Callable that accepts the `settings` dictionary as a single
        positional argument. This function must return a single
        `numpy.ndarray` or a tuple of 1, 2, or 3 elements.
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    rate : numeric
        The refresh rate.
    seed : int
        A random seed.
    kwargs : dict
        Keyword arguments that are kept as attributes. These can be seen
        as specific settings for particular stimuli. These are also
        present in the `settings` attribute.

    Notes
    -----
    The `create_func` can return different values. If it returns a
    `numpy.ndarray`, it is assumed that it corresponds to the
    `signal`. If it returns a tuple of 1 element, this element should
    correspond to the `signal`. If it returns a tuple of 2 elements, the
    first corresponds to the `signal` and the second corresponds to `events`.
    If there are 3 elements, they correspond to the `signal`, `events`, and
    `metadata` respectively.
    """

    def __init__(
        self, create_func,
        *,
        estimator=None,
        rate=None,
        seed=None,
        subsample=None,
        **kwargs
    ):
        assert is_callable(create_func), (
            "Create function must be callable."
        )
        super().__init__(
            create_func=create_func,
            estimator=estimator,
            rate=rate,
            seed=seed,
            subsample=subsample,
            **kwargs
        )

    def create(self):
        output = self.create_func(self.settings)
        if isinstance(output, tuple):
            if len(output) == 1:
                self._signal = output[0]
                self._events = pd.DataFrame()
                self._metadata = {}
            elif len(output) == 2:
                self._signal = output[0]
                self._events = output[1]
                self._metadata = {}
            elif len(output) == 3:
                self._signal = output[0]
                self._events = output[1]
                self._metadata = output[2]
            else:
                raise DreyeError("Output from `create_func` must contain "
                                 "only three tuples maximum, but contains "
                                 f"{len(output)}.")
        else:
            self._signal = output
            self._events = pd.DataFrame()
            self.metadata = {}


class ChainedStimuli:
    """
    Chain multiple stimuli together.

    Parameters
    ----------
    stimuli : list-like
        A list of `dreye.stimuli.BaseStimulus` objects.
    shuffle : bool
        Whether to shuffle the list.
    seed : int
        The seed used for shuffling.
    attrs : object
        Arbitrary object to save with the stimulus set
    """

    def __init__(self, stimuli, shuffle=False, seed=None, attrs=None):

        if isinstance(stimuli, type(self)):
            stimuli = list(stimuli.stimuli)
        elif is_listlike(stimuli):
            assert all(
                isinstance(stim, BaseStimulus)
                for stim in stimuli
            ), "All stimuli must be subclass of BaseStimulus."
            stimuli = list(stimuli)
        else:
            raise DreyeError('Stimuli must be ChainedStimuli or listlike.')

        if seed is not None:
            random.seed(seed)
        if shuffle:
            random.shuffle(stimuli)

        assert all(
            stimuli[0].time_axis == stim.time_axis
            for stim in stimuli
        ), "time axis must be aligned"
        assert all(
            stimuli[0].other_shape == stim.other_shape
            for stim in stimuli
        ), "shapes of stimuli must be the same"
        # assert rates are the same
        assert all(
            stimuli[0].rate == stim.rate
            for stim in stimuli
        ), "stimuli rates must be the same"

        self._stimuli = stimuli
        self.attrs = attrs

    def __len__(self):
        return len(self.stimuli)

    def __iter__(self):
        return iter(self.stimuli)

    def __getitem__(self, key):
        return self.stimuli[key]

    @property
    def rate(self):
        """
        Refresh rate.
        """
        return self.stimuli[0].rate

    @property
    def time_axis(self):
        """
        The time axis.
        """
        return self.stimuli[0].time_axis

    @property
    def stimuli(self):
        """
        The list of stimuli.
        """
        return self._stimuli

    @property
    def events(self):
        """
        A concatenated `events` `pandas.DataFrame`.

        See Also
        --------
        BaseStimulus.events
        """
        events = pd.DataFrame()
        dur = 0.0
        for idx, stim in enumerate(self.stimuli):
            assert CHAINED_STIM_INDEX not in stim.events
            event = stim.events.copy()
            # add duration
            event[DELAY_KEY] += dur
            # add index
            event[CHAINED_STIM_INDEX] = idx
            events = events.append(event, ignore_index=True, sort=True)
            dur += stim.duration
        return events

    @property
    def metadata(self):
        """
        A list of the metadata dictionaries.
        """
        return [
            stim.metadata
            for stim in self.stimuli
        ]

    @property
    def stimulus(self):
        """
        A concatenated stimulus array.
        """
        return np.concatenate([
            stim.stimulus for stim in self.stimuli
        ], axis=self.time_axis)

    @property
    def timestamps(self):
        """
        A concatenated timestamps array.
        """
        return np.concatenate([
            stim.timestamps
            + (0 if idx == 0 else np.cumsum(self.durations)[idx - 1])
            for idx, stim in enumerate(self.stimuli)
        ])

    @property
    def duration(self):
        """
        The complete duration of the stimulus.
        """
        return np.sum(self.durations)

    @property
    def durations(self):
        """
        An array of all the individual durations.
        """
        return asarray([
            stim.duration
            for stim in self.stimuli
        ])

    @property
    def settings(self):
        """
        A list of the settings dictionary.
        """
        return [
            stim.settings
            for stim in self.stimuli
        ]

    def to_dict(self):
        """
        Convert to list of stimuli.
        """
        return {
            'stimuli': self.stimuli,
            'attrs': self.attrs
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create combined stimulus class from list of stimuli
        """
        if isinstance(data, dict):
            return cls(**data)
        return cls(data)

    def save(self, filename):
        """
        Save combined stimulus.
        """
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        """
        Load combined stimulus.
        """
        return read_json(filename)

# have alist of all attributes


@inherit_docstrings
class RandomizeChainedStimuli(ChainedStimuli):
    """
    Randomize a list of stimuli by reordering the concatenated
    `events` dataframe.

    Parameters
    ----------
    stimuli : list-like
        A list of `dreye.stimuli.BaseStimulus` objects.
    shuffle : bool
        Whether the concatenated `events` dataframe.
    seed : int
        The seed used for shuffling.
    """

    def __init__(self, stimuli, shuffle=True, seed=None, attrs=None):
        # initialize chained stimuli
        super().__init__(stimuli, shuffle=False, seed=None, attrs=attrs)

        # dataframe to shuffle
        if shuffle:
            events = super().events
            stimulus = super().stimulus
            # shuffle events
            events = events.sample(
                frac=1, replace=False, random_state=seed
            ).reset_index()
            # get index of first event (offset)
            offset_idx = int(events[DELAY_KEY].min() * self.rate)
            # copy stimulus
            new_stimulus = stimulus.copy()
            # loop over events
            for idx, event in events.iterrows():
                # this should be 0 decimal float anyways
                frames = int(
                    (event[DUR_KEY] + event[PAUSE_KEY]) * self.rate
                )
                # new start and end
                new_start_idx = offset_idx
                new_end_idx = offset_idx + frames
                # old start and end
                old_start_idx = int(event[DELAY_KEY] * self.rate)
                old_end_idx = old_start_idx + frames
                # change stimulus location
                new_stimulus[new_start_idx:new_end_idx] = stimulus[
                    old_start_idx:old_end_idx
                ]

                events.loc[idx, DELAY_KEY] = new_start_idx / self.rate

                offset_idx = new_end_idx

            self._events = events
            self._stimulus = new_stimulus
        else:
            self._events = super().events
            self._stimulus = super().stimulus

    @property
    def events(self):
        return self._events

    @property
    def stimulus(self):
        return self._stimulus

    def to_dict(self):
        """
        A dictionary of all the reshuffled information
        necessary for creating instance.
        """
        return {
            'stimuli': self.stimuli,
            'events': self.events,
            'stimulus': self.stimulus,
            'attrs': self.attrs
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create class from dictionary.
        """
        self = cls(
            data['stimuli'],
            shuffle=False,
            attrs=data.get('attrs', None)
        )
        self._events = data['events']
        self._stimulus = data['stimulus']
        return self
