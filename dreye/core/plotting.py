"""Plotting for Signal Class
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_label_formatted(label):

    label = str(
        label
    ).replace(
        '[', ''
    ).replace(
        ']', ''
    ).replace(
        ' ** ', '^'
    ).replace(
        '*', '\\cdot'
    ).replace('_', '')

    if '/' in label:
        label = "\\frac{" + label.replace('/', '}{', 1) + "}"
        label = label.replace('/', '\\cdot')

    return label


def get_label(units):

    if units.dimensionless:
        label = 'a.u.'
    else:
        label = get_label_formatted(units.dimensionality)
        units = get_label_formatted(units)
        label = f'${label}$ (${units}$)'

    return label


class SignalPlottingMixin:

    # variables preset for class
    _xlabel = None
    _ylabel = None
    _cmap = 'tab10'
    _colors = None

    def _get_cls_vars(
        self,
        cmap, colors, color, xlabel, ylabel
    ):
        if cmap is None:
            cmap = self._cmap
        if colors is None or color is None:
            colors = self._colors
        if xlabel is None:
            xlabel = self._xlabel
        if ylabel is None:
            ylabel = self._ylabel

        if colors is None and color is not None:
            if isinstance(color, str) and self.ndim == 2:
                colors = [color] * self.other_len
            else:
                colors = color

        return cmap, colors, xlabel, ylabel

    def plot(
        self, ax=None,
        labels=False,
        cmap=None, colors=None,
        color=None,
        despine_kwargs={},
        legend_kwargs={},
        xlabel=None,
        ylabel=None,
        **plot_kwargs
    ):
        """
            cmap:
                Color maps sourced from matplotlib.
            colors:
                Can pass list of _colors.
            color:
                Same as colors.
            despine_kwargs: dict-like, optional
                Keyword arguments for seaborn despine function.
            legend_kwargs: dict-like, optional
                Keyword arguments for the matplotlib.pyplot.legend function.
            xlabel: str, optional
                Label for the x-axis. Defaults to units(type), e.g.
                Wavelengths(nm).
            y label: str, optional
                Label for the y-axis. Defaults to units(type), e.g.
                Volts(mV).
            plot_kwargs: dict-like, optional
                Other user-specified keyword values passed to matplotlib.

        """

        cmap, colors, xlabel, ylabel = self._get_cls_vars(
            cmap, colors, color, xlabel, ylabel
        )

        if ax is None:
            ax = plt.gca()

        if self.ndim == 2:
            if colors is None:
                colors = sns.color_palette(cmap, self.other_len)
            else:
                assert len(colors) == self.other_len

            for idx, values in enumerate(
                np.moveaxis(self.magnitude, self.other_axis, 0)
            ):

                ax.plot(
                    self.domain_magnitude,
                    values,
                    label=(self.labels[idx] if labels else None),
                    color=colors[idx],
                    **plot_kwargs
                )

        else:
            ax.plot(
                self.domain_magnitude,
                self.magnitude,
                label=(self.labels if labels else None),
                color=('black' if colors is None else colors),
                **plot_kwargs
            )

        if xlabel is None:
            xlabel = get_label(self.domain_units)
        if ylabel is None:
            ylabel = get_label(self.units)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if labels:
            ax.legend(**legend_kwargs)

        sns.despine(ax=ax, **despine_kwargs)

        return ax

    def relplot(
        self,
        xlabel=None, ylabel=None,
        palette=None,
        **kwargs
    ):
        """aggregate plot using long dataframe and seaborn
        """

        default_kws = dict(
            hue='labels',
            kind='line'
        )
        default_kws.update(kwargs)
        kwargs = default_kws

        default_facet_kws = dict(
            margin_titles=True,
        )
        default_facet_kws.update(kwargs.get('facet_kws', {}))
        kwargs['facet_kws'] = default_facet_kws

        palette, _, xlabel, ylabel = self._get_cls_vars(
            palette, None, None, xlabel, ylabel
        )

        data = self.to_longframe()

        g = sns.relplot(
            data=data,
            x='domain',
            y='values',
            palette=palette,
            **kwargs
        )

        if xlabel is None:
            xlabel = get_label(self.domain_units)
        if ylabel is None:
            ylabel = get_label(self.units)

        g.set_xlabels(xlabel)
        g.set_ylabels(ylabel)

        return g
