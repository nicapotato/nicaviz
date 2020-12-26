import numpy as np
import pandas as pd


# Data Exploration
def describe(df, value_count_n=5):
    """
    Custom Describe Function - More Tailored to categorical type variables..
    """
    unique_count = []
    for x in df.columns:
        unique_values_count = df[x].nunique()
        value_count = df[x].value_counts().iloc[:5]

        value_count_list = []
        value_count_string = []

        for vc_i in range(0, value_count_n):
            value_count_string += ["ValCount {}".format(vc_i + 1),
                                   "Occ"]
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
    return pd.DataFrame(unique_count,
                        columns=["Column", "Unique", "Missing", "dtype"
                                 ] + value_count_string
                        ).set_index("Column")
