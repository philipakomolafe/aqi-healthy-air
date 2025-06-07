import pandas as pd
import os
import pickle

def cache_features(data: pd.DataFrame, cache_path: str):
    """
    Cache intermediate features using pickle.
    """
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cached_features(cache_path: str) -> pd.DataFrame:
    """
    Load cached features from pickle file.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file {cache_path} not found.")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
