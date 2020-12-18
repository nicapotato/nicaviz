import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import itertools

def big_bar_cloud(plot_df, plt_set, x_var, columns, figsize, custom_palette = sns.color_palette("Paired")):
    """
    Iteratively Plot BarPlots
    """
    palette = itertools.cycle(custom_palette)
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f, ax = plt.subplots(rows, columns, figsize = figsize)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            col = plt_set[i]
            sns.barplot(data=plot_df,
                        x=x_var,
                        y=col,
                        ax=ax,
                        color=next(palette),
                        alpha=.8)
            ax.set_title("{} by {}".format(col, x_var))
        else:
            ax.axis('off')
    plt.tight_layout(pad=2)



def rank_correlations(df, figsize=(12,20), n_charts = 18, polyorder = 2, custom_palette = sns.color_palette("Paired", 5)):
    # Rank Correlations
    palette = itertools.cycle(custom_palette)
    continuous_rankedcorr = (df
                             .corr()
                             .unstack()
                             .drop_duplicates().reset_index())
    continuous_rankedcorr.columns = ["f1","f2","Correlation Coefficient"]
    continuous_rankedcorr['abs_cor'] = abs(continuous_rankedcorr["Correlation Coefficient"])
    continuous_rankedcorr.sort_values(by='abs_cor', ascending=False, inplace=True)

    # Plot Top Correlations
    top_corr = [(x,y,cor) for x,y,cor in list(continuous_rankedcorr.iloc[:, :3].values) if x != y]
    f, axes = plt.subplots(int(n_charts/3),3, figsize=figsize, sharex=False, sharey=False)
    row = 0
    col = 0
    for (x,y, cor) in top_corr[:n_charts]:
        if col == 3:
            col = 0
            row += 1
        g = sns.regplot(x=x, y=y, data=df, order=polyorder, ax = axes[row,col], color=next(palette))
        axes[row,col].set_title('{} and {}'.format(x, y))
        axes[row,col].text(0.18, 0.93,"Cor Coef: {:.2f}".format(cor),
                           ha='center', va='center', transform=axes[row,col].transAxes)
        col += 1
    plt.tight_layout(pad=0)
    plt.show()