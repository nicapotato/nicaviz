import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import itertools
from wordcloud import WordCloud
from itertools import combinations

def big_boxplotter(plot_df, plt_set, columns, figsize, hue = None, plottype='kde',
                   custom_palette = sns.color_palette("Dark2", 15), quantile = .99):
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    palette = itertools.cycle(custom_palette)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            cont_col = plt_set[i]
            if hue:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) &
                                      (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      [cont_col, hue]]
                if plottype == 'box':
                    sns.boxplot(data=plt_tmp, x=cont_col, y=hue, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    for h in plt_tmp.dropna()[hue].value_counts()[:5].index:
                        c = next(palette)
                        sns.distplot(plt_tmp.loc[plt_tmp[hue] == h,cont_col], bins=10, kde=True, ax=ax,
                                     kde_kws={"color": c, "lw": 2, "label":h}, color=c)
                    ax.set_ylabel("Density Occurence")
            else:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) &
                                      (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      cont_col].astype(float)
                if plottype == 'box':
                    sns.boxplot(plt_tmp, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    sns.distplot(plt_tmp, bins=10, kde=True, ax=ax,
                                 kde_kws={"color": "k", "lw": 2}, color=next(palette))
                    ax.set_ylabel("Density Occurence")
            ax.set_title("{} - {:.0f} Missing - {:.2f} Max".format(cont_col.title(),
                                                                   plot_df[cont_col].isnull().sum(), plot_df[cont_col].max()))
            ax.set_xlabel("Value")

        else:
            ax.axis('off')

    plt.tight_layout(pad=1)