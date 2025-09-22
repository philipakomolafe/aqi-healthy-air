import pandas as pd
import numpy as np

# Feature engineering for AQI dataset
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering steps to AQI data.
    Args:
        df (pd.DataFrame): Cleaned AQI data
    Returns:
        pd.DataFrame: DataFrame with new features
    """
    df = df.copy()

    # 1. Time-based features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['weekday'] = df['timestamp'].dt.weekday
        df['season'] = df['month'] % 12 // 3 + 1  # 1:Winter, 2:Spring, 3:Summer, 4:Fall


    # 2. Interaction features
    # Example: PM2.5/PM10 ratio, temp*humidity
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

    # 4. Domain-specific features
    # Assuming 'aqi' is the categorized column (1-5)
    if 'aqi' in df.columns:
        # Rolling mean and max of AQI category over 24h window
        df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
        df['aqi_rolling_max_24h'] = df['aqi'].rolling(window=24, min_periods=1).max()
        # Count of hours in last 24h where AQI was above a threshold (e.g., >3)
        df['aqi_above_3_count_24h'] = df['aqi'].rolling(window=24, min_periods=1).apply(lambda x: (x > 3).sum(), raw=True)
        # Difference between current and previous AQI category (trend)
        df['aqi_trend'] = df['aqi'].diff().fillna(0)

    return df

if __name__ == "__main__":
    # Example usage
    from data_cleaner import load_data, clean_data
    from utils import config_loader

    config = config_loader()
    raw = load_data()
    clean = clean_data(raw)
    features = feature_engineering(clean)
    print(features.head())
    # Save features.
    
