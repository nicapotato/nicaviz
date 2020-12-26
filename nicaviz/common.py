def prepare_title(string):
    return string.replace("_", " ").title()

def pd_continuous_null_and_outliers(df, col, upper_percentile, lower_percentile = None):
    df = df.loc[df[col].notnull(),:]
    upper = df[col].quantile(upper_percentile/100, interpolation="lower")
    if lower_percentile:
        lower = df[col].quantile(lower_percentile/100, interpolation="lower")
        return df.loc[(df[col] <= upper) & (df[col] >= lower),:]
    else:
        return df.loc[(df[col] <= upper),:]

def pd_categorical_reduce(df, col, top_n_categories, strategy):
    topcat = df[col].value_counts().index[:top_n_categories]
    if strategy == "as other":
        df.loc[~df[col].isin(topcat),col] = "Other"
    elif strategy == "exclude":
        df = df.loc[df[col].isin(topcat), :]
    else:
        raise "Invalid Strategy for pd_categorical_reduce()"
    return df

