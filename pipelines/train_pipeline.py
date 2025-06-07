# NOTE the training pipeline contains:
# 1. call splited files from `data/processed` folder.
# 2. model training function.
# 3. model evaluation function.
# 4. model artifact loggng using Neptune Utils.

import os
import sys
import pandas as pd
from loguru._logger import Logger
from pathlib import Path

# Adding project root directory to python path.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.train import train_model, retrain_model
from src.utils import config_loader, read_processed_data, setup_logger, get_logger


def run(
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        log: Logger, 
        model=None, 
        config=None, 
        re_train=False
        ):
    try:
        if re_train:
            return retrain_model(train_data, val_data, log, model, config)
        else:
            return train_model(train_data, val_data, log, config)
    except Exception as e:
        log.error(f"Error encountered: {e}")


# Defining callable to run the Training Pipeline... 
def main():
    # Init Logger & Yaml configurations.
    config = config_loader()
    setup_logger(level='INFO')
    log = get_logger("Train Pipeline")

    # Initialize datasets paths - [TRAIN, VAL, TEST].
    train_path = f"{config['dataset']['processed']['train']}/aqi_train_data_v1.csv"
    val_path = f"{config['dataset']['processed']['val']}/aqi_val_data_v1.csv"
    test_path = f"{config['dataset']['processed']['test']}/aqi_test_data_v1.csv"

    # Define the dataframes.
    train_df = read_processed_data(train_path, log)
    val_df = read_processed_data(val_path, log)
    test_df = read_processed_data(test_path, log) 

    # Run training instance for best validation model.
    log.info("Running training instance begins...")
<<<<<<< HEAD
    best_model = run(train_data=train_df,
                    val_data=val_df,
                    log=log, # type: ignore
                    config=config)
    log.success("Training instance complete...")

    # Retrain the model with train and val datasets.
    log.info('Begin Model retraining...')
    _ = run(train_data=train_df,
            val_data=val_df,
            log=log,        # type: ignore
            model=best_model,
            config=config,
            re_train=True)
    log.success("Re-training instance completed...")
=======
    _ = run(train_data=train_df,
            val_data=val_df,
            log=log, # type: ignore
            config=config
            )
    log.info("Saving Model to Model Registry...")
    log.success("Training instance complete...")
>>>>>>> ccf4cd5fac4c8873fa7ca381663338d92e698d84

        
    