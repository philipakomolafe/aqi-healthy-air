# NOTE: utils.py contains all the helper functions ike logger for loguru and config loader.
import re
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

config_path = Path(__file__).parent.parent / "config" / "config.yaml"
def config_loader(path=config_path):
    "Helper function to load the config file."
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Convet  from datetime to integer (secs).
def to_unix_timestamp(datetime_obj) -> int:
    return int(time.mktime(datetime_obj.timetuple()))


# TODO: Define the Logger utils to log file processes..
def setup_logger(log_file: str = 'C:/Users/user/Desktop/startups/aqi_mvp/logs/pipeline.log', level: str = "DEBUG"):
    """
    Configure the Loguru logger.
    """
    # Remove existent loggers
    logger.remove()

    # Logging to console.
    logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    # File logging.
    logger.add(log_file, rotation='1 MB', retention="7 days", compression='zip', level=level, enqueue=True, backtrace=True, 
                diagnose=True)

def get_logger(name: str | None = None):
    """
    Create ready to use logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger

def read_processed_data(path: str, log) -> pd.DataFrame:
    """
    Read the CSV file and return a DataFrame.
    """
    try:
        data = pd.read_csv(path)
        log.info(f"Data read successfully from {path}")
        return data
    except FileNotFoundError:
        log.error(f"File not found: {path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        log.error(f"File is empty: {path}")
        return pd.DataFrame()

# TODO: Define the prefect orchestration utilities..

def categorize_aqi(aqi: float):
    if aqi <= 0 :
        return "Invalid AQI"
    elif aqi <= 1:
        return "Excellent (No health risk)"
    elif aqi <= 2:
        return "Good (Low health risk)"
    elif aqi <= 3:
        return "Moderate (Sensitive groups may experience health effects)"
    elif aqi <= 4:
        return "Poor (Unhealthy for sensitive groups)"
    elif aqi <= 5:
        return "Very Poor (Unhealthy for all)"
    else:
        return "Hazardous (Serious health effects for all)" 

def color_aqi(aqi: float):
    if aqi <= 0:
        return "Invalid"
    elif aqi <= 1:
        return "Green"
    elif aqi <= 2:
        return "Yellow"
    elif aqi <= 3:
        return "Orange"
    elif aqi <= 4:
        return "Red"
    elif aqi <= 5:
        return "Purple"
    else:
        return "Pale"

def postprocess_predictions(predictions: np.ndarray) -> pd.DataFrame:
    """
    postprocess the predictions for health related user-friendly AQI values.
    """
    df = pd.DataFrame({"predicted_aqi": predictions})
    # categorize the AQI level as good, bad or e.t.c for health conscious population.
    df['aqi_category'] = df['predicted_aqi'].apply(categorize_aqi)
    # Air quality health color to give patient deeper view on the AQI status..
    df['aqi_health_color'] = df['predicted_aqi'].apply(color_aqi) 
    return df

def fetch_current_data(config):
    try:
        start_ts = int(time.mktime(datetime.today().timetuple()))
        end_ts = int(time.mktime((datetime.today() + timedelta(hours=1)).timetuple()))
        params=config['openweather']['params']
        params['start']=start_ts
        params['end']=end_ts

        response = requests.get(
            url=config['openweather']['base_url'],
            params=params
        )
        # Similarly checking for HTTPerror. Totology anyways.
        response.raise_for_status()
        # Dump data into JSON format.
        data = response.json()

        # Return the features to compute the AQI value.
        return {
                'no': data['list'][0]['components']['no'],
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'co': data['list'][0]['components']['co'],
                'no2': data['list'][0]['components']['no2'],
                'so2': data['list'][0]['components']['so2'],
                'o3': data['list'][0]['components']['o3'],
                'nh3': data['list'][0]['components']['nh3'],
                'timestamp': datetime.utcfromtimestamp(data['list'][0]['dt']),
                'aqi': data['list'][0]['main']['aqi']
                }
                # 'timestamp', 'aqi', 'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3', 'nh3'
    
    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {e}"
    except Exception as e:
        return f"Other Errors occurred -> {e}"


def select_best_model(model_directory: Path, weight=(0.6, 0.4)) -> tuple[str, float, float, float]:
    """Select the best model based off the weighted sum of the Balanced accuracy and Macro ROC-AUC values..."""
    best_model_name = None
    best_roc = -1
    best_acc = -1
    best_score = -1

    # Define search pattern
    pattern = re.compile(r"acc_(\d+\.\d+)_roc_(\d+\.\d+)\.pkl")
    # Iterate through all files in the directory
    for file in os.listdir(model_directory):
        match = pattern.search(file)
        if match:
            acc = float(match.group(1))
            roc = float(match.group(2))

            # weighted sum of balanced accuracy and macro ROC
            w1, w2 = weight
            score = (w1 * roc) + (w2 * acc)

            if score > best_score:
                best_score = score
                best_model_name = f"{model_directory}\{file}"
                best_acc = acc
                best_roc = roc

    return best_model_name, best_acc, best_roc, best_score



if __name__ == "__main__":
    # config = config_loader()
    # print(fetch_current_data(config))

    pass