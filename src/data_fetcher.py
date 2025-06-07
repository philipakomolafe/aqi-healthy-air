#  `data_loader.py` 
# NOTE: for loading `Air Quality index parameters from OPENWEATHER API`

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import requests

from pydantic import BaseModel
from src.utils import config_loader, to_unix_timestamp, setup_logger, get_logger
from typing import List
# from dotenv import load_dotenv


class MainAQI(BaseModel):
    aqi: float    # 1 -> Good  &  5 -> Bad

class Components(BaseModel):
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float

class AQIParams(BaseModel):
    main: MainAQI
    components: Components
    dt: int # timestamp

class AQIResponse(BaseModel):
    list: List[AQIParams]

# Init Logger Configuration
setup_logger(level='INFO')
log = get_logger()

# Init yaml Config instance
config = config_loader()

# Fetch AQI Data.
def fetch_data(config):
    # Datalist
    data_list = []
    interval_days = 60  # 2-months per fetch cycle.

    # Defne the start and end date for FETCHING the data. 
    start_date = datetime(2016, 1, 1)
    end_date = datetime.today()

    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=interval_days), end_date)

        start_ts = to_unix_timestamp(current_start)
        end_ts = to_unix_timestamp(current_end)

        # Update the config with the new start and end timestamps.
        config['openweather']['params']['start'] = start_ts
        config['openweather']['params']['end'] = end_ts

        # fetch the Base url and url parameters for openweather
        base_url = config["openweather"]["base_url"]
        params = config["openweather"]["params"]

        try:
            response = requests.get(base_url, params, timeout=10)
            response.raise_for_status()
            data_raw = response.json()

            # Validate incoming data using pydantic.
            validated_data = AQIResponse(**data_raw)
            for entry in validated_data.list:
                data_list.append({
                    "timestamp": datetime.utcfromtimestamp(entry.dt),
<<<<<<< HEAD
                    "aqi": int(entry.main.aqi),
=======
                    "aqi": entry.main.aqi,
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84
                    "no": entry.components.no,  # recently added to features..
                    "pm2_5": entry.components.pm2_5,
                    'pm10': entry.components.pm10,
                    "co": entry.components.co,
                    "no2": entry.components.no2,
                    "so2": entry.components.so2,
                    "o3": entry.components.o3,
                    "nh3": entry.components.nh3
                    })
                
            log.info(f"✅ Data pulled for {current_start.date()} - {current_end.date()}")

        except requests.exceptions.Timeout:
            log.info("⚠ Timeout error! Skipping this batch.")
        
        except requests.exceptions.RequestException as e:
            log.info(f"❌ Request failed: {e}")

        time.sleep(5)
        current_start = current_end + timedelta(days=1)
        
    # Save to CSV.
    if data_list:

        # Define the data path.
        raw_path = config["dataset"]["raw"]
        filename = f'aqi_data_{start_date.year}-{end_date.year}.csv'
        data_path = os.path.join(raw_path, filename)

        df = pd.DataFrame(data_list)
        df.to_csv(data_path, index=False)
        log.info(f"✔ AQI Data saved to `aqi_data_{start_date.year}-{end_date.year}.csv`")

    else:
        log.info("⚠ No data Was collected!!")



if __name__ == "__main__":
    # Fetch data using the access given by `config`. 
    # config = config_loader()
    fetch_data(config=config)