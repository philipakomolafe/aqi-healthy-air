# NOTE the feature pipeline contains the following:
# 1. Data Cleaning.
# 2. feature extraction/engineering.
 
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import config_loader, setup_logger, get_logger
from src.data_fetcher import fetch_data
from src.data_cleaner import load_data, clean_data
from src.feature_engineering import feature_engineering
from src.feature_selection import correlation_analysis, feature_importance
from src.data_splitter import time_series_split
from src.feature_storage import save_features
<<<<<<< HEAD
# from src.feature_cache import cache_features
=======
from src.feature_cache import cache_features
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84


# Defining callable for Feature Pipeline..
def main():
    # Initialize the logger utilities
    setup_logger(level="INFO")
    log = get_logger("Feature Pipeline")

    log.info("Starting feature pipeline...")
    # Load configuration
    config = config_loader()
    
    log.info("Data downloading from API...")
    fetch_data(config=config)
<<<<<<< HEAD
    log.success("Data download completed...")
=======
    log.info("Data download completed...")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    log.info("Data Loading & Cleaning...")
    # 1. Data Loading & Cleaning
    raw = load_data()
    clean = clean_data(raw)
<<<<<<< HEAD
    log.success("Data Loading & Cleaning Completed.")
=======
    log.info("Data Loading & Cleaning Completed.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    log.info("Feature Engineering...")
    # 2. Feature Engineering
    features = feature_engineering(clean)
<<<<<<< HEAD
    log.success("Feature Engineering Completed.")
=======
    log.info("Feature Engineering Completed.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    log.info("Feature Selection...")
    # 3. Feature Selection
    log.info("Performing correlation analysis...")
    # Remove highly correlated features
    reduced_features, corr_matrix, dropped = correlation_analysis(features, threshold=0.9)
    log.info(f"\nDropped highly correlated features: \n{dropped}")
<<<<<<< HEAD
    log.success("Correlation analysis completed.")
=======
    log.info("Correlation analysis completed.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    # TODO: Save the correlation matrix for reference

    log.info("Feature importance analysis...")
    # Feature importance (example: using 'aqi' as target)
    importances = feature_importance(reduced_features, target_col='aqi', task='classification', n_top=20)
    log.info(f"Top feature importances: \n{importances}")
<<<<<<< HEAD
    log.success("Feature importance analysis completed.")
=======
    log.info("Feature importance analysis completed.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    # 4. Data Splitting (time-based)
    log.info("Splitting data into train, validation, and test sets...")
    train, val, test = time_series_split(features, time_col='timestamp',
                              train_size=0.7, val_size=0.15)
    log.info(f"\nTrain shape: \n{train.shape}\n Val shape: {val.shape}\n Test shape: {test.shape}")
<<<<<<< HEAD
    log.success("Data splitting completed.")
=======
    log.info("Data splitting completed.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

    # 5. Feature Storage (with versioning)
    log.info("Saving features to CSV files...")
    save_features(data=train, filename="aqi_train_data.csv", output_dir=config["dataset"]["processed"]['train'], version='1')
    save_features(data=val, filename="aqi_val_data.csv", output_dir=config["dataset"]["processed"]['val'], version='1')
    save_features(data=test, filename="aqi_test_data.csv", output_dir=config["dataset"]["processed"]['test'], version='1')
<<<<<<< HEAD
    log.success("Features saved successfully.")
=======
    log.info("Features saved successfully.")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84



