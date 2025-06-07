import pandas as pd
from typing import Tuple

def time_series_split(df: pd.DataFrame, time_col: str, train_size: float = 0.7, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets based on time for time series data.
    Args:
        df (pd.DataFrame): Input dataframe
        time_col (str): Name of the datetime column
        train_size (float): Proportion for training set
        val_size (float): Proportion for validation set
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    df_sorted = df.sort_values(by=time_col)
    n = len(df_sorted)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test
