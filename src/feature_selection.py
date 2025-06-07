import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def correlation_analysis(df: pd.DataFrame, threshold: float = 0.9):
    """
    Remove highly correlated features above a given threshold.
    Returns the dataframe with reduced features and a correlation matrix.
    """
    df = df.drop(columns=['timestamp'])
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    reduced_df = df.drop(columns=to_drop)
    return reduced_df, corr_matrix, to_drop

def feature_importance(df: pd.DataFrame, target_col: str, task: str = 'classification', n_top: int = 20):
    """
    Calculate feature importances using RandomForest.
    Returns a sorted list of top features.
    """
    X = df.drop(columns=[target_col], axis=1)
    y = df[target_col]
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(n_top)