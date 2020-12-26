import math

import matplotlib.pyplot as plt


class Plotgrid:
    """
    Class to plot matplotlib objects in a grid
    """
    def __init__(self, plt_set, columns=2, figsize=None):
        self.columns = columns
        self.rows = self._calc_rows(len(plt_set), columns)
        self.n_plots = self.rows * self.columns
        self.plt_set = plt_set
        self.figsize = figsize if figsize else self._estimate_figsize(self.columns, self.rows)

    def _calc_rows(self, n_plots, columns):
        return math.ceil(n_plots / columns)

    def _estimate_figsize(self, columns, rows):
        figsize = [columns * 5, rows * 4, ]
        return figsize

    def build(self, func, df, **kwargs):
        f, ax = plt.subplots(self.rows,
                             self.columns,
                             figsize=self.figsize)
        for i in range(0, self.n_plots):
            ax = plt.subplot(self.rows, self.columns, i + 1)
            if i < len(self.plt_set):
                func(self.plt_set[i], ax, df, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout(pad=1)
