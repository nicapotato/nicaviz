import pandas as pd
import numpy as np
from numpy import random
import itertools

import math
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def pd_continuous_null_and_outliers(df, col, upper_percentile, lower_percentile=None):
    df = df.loc[df[col].notnull(), :]
    upper = df[col].quantile(upper_percentile/100, interpolation="lower")
    if lower_percentile:
        lower = df[col].quantile(lower_percentile/100, interpolation="lower")
        return df.loc[(df[col] <= upper) & (df[col] >= lower), :]
    else:
        return df.loc[(df[col] <= upper), :]


def pd_categorical_reduce(df, col, top_n_categories, strategy):
    topcat = df[col].value_counts().index[:top_n_categories]
    if strategy == "as other":
        df.loc[~df[col].isin(topcat), col] = "Other"
    elif strategy == "exclude":
        df = df.loc[df[col].isin(topcat), :]
    else:
        raise "Invalid Strategy for pd_categorical_reduce()"
    return df

# Data Exploration


def describe_categorical(df, value_count_n=5):
    """
    Custom Describe Function for categorical variables
    """
    unique_count = []
    for x in df.columns:
        unique_values_count = df[x].nunique()
        value_count = df[x].value_counts().iloc[:5]

        value_count_list = []
        value_count_string = []

        for vc_i in range(0, value_count_n):
            value_count_string += ["ValCount {}".format(vc_i + 1), "Occ"]
            if vc_i <= unique_values_count - 1:
                value_count_list.append(value_count.index[vc_i])
                value_count_list.append(value_count.iloc[vc_i])
            else:
                value_count_list.append(np.nan)
                value_count_list.append(np.nan)

        unique_count.append([x,
                             unique_values_count,
                             df[x].isnull().sum(),
                             df[x].dtypes] + value_count_list)

    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count, columns=["Column", "Unique", "Missing", "dtype"] + value_count_string).set_index("Column")


@pd.api.extensions.register_dataframe_accessor("nica")
class NicaAccessor(object):
    """
    Class to plot matplotlib objects in a grid
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def rank_correlations_plots(self, continuouscols, n, columns=3, polyorder=2, figsize=None, palette=None):
        self.rank_df = self.get_corr_matrix(self._obj[continuouscols])
        self.plt_set = [(x, y, cor) for x, y, cor in list(
            self.rank_df.iloc[:, :3].values) if x != y][:n]
        self._gridparams(len(self.plt_set), columns, figsize, palette)

        f, ax = plt.subplots(self.rows, self.columns, figsize=self.figsize)
        # func, fkwargs = self._get_plot_func(plottype)
        for i in range(0, self.n_plots):
            ax = plt.subplot(self.rows, self.columns, i + 1)
            if i < len(self.plt_set):
                self.regplot(self.plt_set[i], ax, self._obj, polyorder=2)
            else:
                ax.axis('off')
        plt.tight_layout(pad=1)

    def mass_plot(self, plt_set, plottype, columns=2, figsize=None, palette=None, **kwargs):
        self._gridparams(len(plt_set), columns, figsize, palette)
        self.plt_set = plt_set

        f, ax = plt.subplots(self.rows, self.columns, figsize=self.figsize)
        for i in range(0, self.n_plots):
            ax = plt.subplot(self.rows, self.columns, i + 1)
            if i < len(self.plt_set):
                func, fkwargs = self._get_plot_func(plottype)
                func(self.plt_set[i], ax, self._obj, **kwargs, **fkwargs)
            else:
                ax.axis('off')
        plt.tight_layout(pad=1)

    def _gridparams(self, plotlen, columns=2, figsize=None, palette=None):
        # Dimensions
        self.columns = columns
        self.rows = self._calc_rows(plotlen, columns)
        self.n_plots = self.rows * self.columns
        self.figsize = figsize if figsize else self._estimate_figsize(
            self.columns, self.rows)

        # Colors
        self.palette = palette if palette else sns.color_palette("Paired")[
            1::2]
        self.iti_palette = itertools.cycle(self.palette)

    def _calc_rows(self, n_plots, columns):
        return math.ceil(n_plots / columns)

    def _estimate_figsize(self, columns, rows):
        figsize = [columns * 5, rows * 4]
        return figsize

    def categorical_describe(self):
        return describe_categorical(self._obj)

    def _get_plot_func(self, plottype):
        switcher = {
            'boxplot': [self.multi_plot, {'plottype': plottype}],
            'countplot': [self.multi_plot, {'plottype': plottype}],
            'distplot': [self.custom_distplot, {}],
            'wordcloud': [self.plot_cloud, {}],
            'bar': [self.single_bar, {}],
            'ts_resample': [self.ts_resample, {}],
            'ts_rolling': [self.ts_rolling, {}]
        }
        # Get the function from switcher dictionary
        func, fkwargs = switcher.get(plottype, lambda: "Invalid Plottype")
        return func, fkwargs

    def multi_plot(self, col, ax, df, plottype, hue=None, top_n=10):
        order = df[col].value_counts().index[:top_n]
        clean_col_name = self.prepare_title(col)
        missing = df[col].isnull().sum()

        if hue:
            pkwarg = {"palette": self.palette}
            clean_hue_name = self.prepare_title(hue)
            ax.set_title(
                "{} by {} - {:.0f} Missing".format(clean_col_name, clean_hue_name, missing))
        else:
            pkwarg = {"color": next(self.iti_palette)}
            ax.set_title("{} - {:.0f} Missing".format(clean_col_name, missing))

        if plottype == "countplot":
            pkwarg['alpha'] = 0.5
            pkwarg['edgecolor'] = "black"
            pkwarg['linewidth'] = 1
            pkwarg['order'] = order

            if hue:
                pkwarg['hue'] = hue
            sns.countplot(data=df, y=col, ax=ax, **pkwarg)

        if plottype == "boxplot":
            if hue:
                pkwarg["y"] = hue
            sns.boxplot(data=df, x=col, ax=ax, **pkwarg)

        ax.set_ylabel(clean_col_name)
        ax.set_xlabel("Count")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def custom_distplot(self, col, ax, df, hue=None, top_n=10):
        valmin, valmax = df[col].min(), df[col].max()
        clean_col_name = self.prepare_title(col)
        missing = df[col].isnull().sum()

        assert hue != col, "Hue cannot equal Col"

        if hue:
            tmp = df.loc[:, [col, hue]].copy()
            hue_cats = tmp[hue].value_counts().index[:top_n]
            clean_huecol_name = self.prepare_title(hue)
            for h in hue_cats:
                pdf = tmp.loc[tmp[hue] == h, col]
                pal = next(self.iti_palette)
                sns.distplot(pdf, ax=ax, color=pal, kde_kws={
                             "color": pal, "lw": 2}, label=str(h))
            ax.set_title(
                "{} by {} - {:.0f} Missing".format(clean_col_name, clean_huecol_name, missing))
            ax.legend()
        else:
            sns.distplot(df[col], ax=ax, color=next(
                self.iti_palette), kde_kws={"color": "k", "lw": 2})
            ax.set_title("{}".format(clean_col_name))

        ax.set_xlim(valmin, valmax)
        ax.set_xlabel(clean_col_name)
        ax.set_ylabel("Density")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, lw=1, ls='--', c='.75')

    def clean_str_arr(self, series):
        if series.shape[0] == 0:
            return "EMPTY"
        else:
            series = series.dropna().astype(str).str.lower().str.replace("none", "").str.title()
            return " ".join(series)

    def plot_cloud(self, col, ax, df, cmap="plasma"):
        missing = df[col].isnull().sum()
        clean_col_name = self.prepare_title(col)
        string = self.clean_str_arr(df[col].copy())
        title = "{} Wordcloud - {:.0f} Missing".format(clean_col_name, missing)

        wordcloud = WordCloud(width=800, height=500,
                              collocations=True,
                              background_color="black",
                              max_words=100,
                              colormap=cmap).generate(string)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=18)
        ax.axis('off')

    def single_bar(self, col, ax, df, x_var):
        clean_col_name, clean_x_var_name = self.prepare_title(
            col), self.prepare_title(x_var)
        missing = df[col].isnull().sum()
        sns.barplot(data=df, x=x_var, y=col, ax=ax, color=next(
            self.iti_palette), linewidth=1, alpha=.8)
        ax.set_title(
            "{} by {} - Missing {:.0f}".format(clean_col_name, clean_x_var_name, missing))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def ts_rolling(self, col, ax, df, x_var, rolling=False, r=0):
        clean_col_name, clean_x_var_name = self.prepare_title(
            col), self.prepare_title(x_var)

        if rolling:
            ts_plot = df.loc[:, [col, x_var]].set_index(x_var)
            ts_plot = ts_plot.rolling(r).mean()

        ts_plot.dropna().plot(ax=ax, color=next(self.iti_palette), alpha=1, lw=2)

        ax.set_xlabel(clean_col_name)
        ax.set_ylabel(clean_x_var_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, lw=1, ls='--', c='.75')

    def ts_resample(self, col, ax, df, resample=False, resample_interval="1D"):
        clean_col_name, clean_x_var_name = self.prepare_title(
            col), self.prepare_title(col)

        if resample:
            ts_plot = df[col].reset_index().copy()
            ts_plot['placeholder'] = "N/A"
            ts_plot = ts_plot.set_index(col)['placeholder'].resample(
                resample_interval).count()

        ts_plot.dropna().plot(ax=ax, color=next(self.iti_palette), alpha=1, lw=2)

        ax.set_xlabel(clean_col_name)
        ax.set_ylabel(clean_x_var_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, lw=1, ls='--', c='.75')

    def prepare_title(self, string):
        return string.replace("_", " ").title()

    def get_corr_matrix(self, df):
        continuous_rankedcorr = df.corr().unstack().drop_duplicates().reset_index()
        continuous_rankedcorr.columns = ["f1", "f2", "Correlation Coefficient"]
        continuous_rankedcorr['abs_cor'] = abs(
            continuous_rankedcorr["Correlation Coefficient"])
        continuous_rankedcorr.sort_values(
            by='abs_cor', ascending=False, inplace=True)

        return continuous_rankedcorr

    def regplot(self, xy, ax, df, polyorder):
        x, y, cor = xy
        g = sns.regplot(x=x, y=y, data=df, order=polyorder,
                        ax=ax, color=next(self.iti_palette))
        ax.set_title('{} and {}'.format(x, y))
        ax.text(0.18, 0.93, "Cor Coef: {:.2f}".format(
            cor), ha='center', va='center', transform=ax.transAxes)
