import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import itertools
from wordcloud import WordCloud
from itertools import combinations

def big_word_cloud(plot_df, plt_set, columns, figsize, cmap = "plasma"):
    """
    Iteratively Plot WordClouds
    """
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            str_col = plt_set[i]
            string = " ".join(plot_df.loc[plot_df[str_col].notnull(),str_col] \
                              .astype(str).str.lower().str.replace("none", "").str.title())
            string += 'EMPTY'
            ax = plt.subplot(rows, 2, i+1)
            plot_cloud(string, ax, title = "{} - {:.0f} Missing".format(
                str_col.title(), plot_df[str_col].isnull().sum()), cmap = cmap)
        else:
            ax.axis('off')
    plt.tight_layout(pad=0)