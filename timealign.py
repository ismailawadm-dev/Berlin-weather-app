import pandas as pd

def to_3h_bins(times):
    return pd.to_datetime(times).dt.floor("3H")
