import pandas as pd

def meta_text_features(df: pd.DataFrame, col):
    df[col] = df[col].astype(str)
    df[col + '_num_words'] = df[col].apply(lambda comment: len(comment.split())) # Count number of Words
    df[col + '_num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df[col + '_words_vs_unique'] = df[col+'_num_unique_words'] / df[col+'_num_words'] * 100 # Count Unique Words

    return df