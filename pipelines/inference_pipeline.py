# NOTE: The inference pipeline is responsible for loading the model and making predictions.
# NOTE: the inference pipeline also contains the following:
# 1. Load Configurations.
# 2. Load Model from `Model Registry`
# 3. Prediction function.
# 4. Return Output to the API.


import sys
import uvicorn
import pandas as pd
from pathlib import Path
from typing import Union, List
from pydantic import BaseModel
from datetime import datetime 
from fastapi import FastAPI, HTTPException

# Adding the project root.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model_loader import load_model
# from src.data_fetcher import AQIResponse, Components
from src.utils import config_loader, setup_logger, get_logger, postprocess_predictions, fetch_current_data, read_processed_data
from src.feature_engineering import feature_engineering


# Initialize config and logger.
config = config_loader()
setup_logger(level="INFO")
log = get_logger("Inference Pipeline")


# Define model for API response.
class PredictionOutput(BaseModel):
        location: str
        predicted_aqi: int
        advice: str
        color: str
        timestamp: datetime
  


# Defining callable to run the Inference Pipeline..
def create_app():
    # Define the model path from the config.
    model_path = f"{config['model_registry']['model_path']}/xgb_acc_0.845_roc_0.996.pkl"
    # Define test data path.
    # test_path = f"{config['dataset']['processed']['test']}/aqi_test_data_v1.csv"

    # Instantiate the model loader.
    model = load_model(model_path=model_path, log=log)   # type: ignore
    log.success("Model loaded successfully for inference...")

    # Init App instance.
    app = FastAPI(
        title="AIR QUALITY INDEX SYSTEM.",
        description="A system designed to predict the air quality of any city - Akure."
        )

    @app.get("/")
    def root(summary="Root endpoint for AQI web API"):
        return {'message': "Welcome to the Air Quality prediction system..."}


    @app.get('/aqi/online-prediction', response_model=PredictionOutput)
    def online_predict():
        # Get the current AQI features :->.
        aqi_features = fetch_current_data(config=config)
        df = pd.DataFrame([aqi_features], index=[0])    # Here: adding index to the dict helps create a 1-row index for the dataframe.
        # Apply feature engineering..
        feature_eng = feature_engineering(df)
        # Drop unnecessary columns.
        feature_eng = feature_eng.drop(columns=['timestamp', 'aqi', 'no'], errors='ignore')
        # make prediction using the loaded model..
        prediction = model.predict(feature_eng.values)  
        prediction = prediction + 1  # Adjusting the prediction to match the AQI scale (1-5).
        # Postprocess the predictions.

        # json output. 
        return {
            'location': "Akure, Nigeria",
            'predicted_aqi': int(prediction[0]),
            'advice': postprocess_predictions(prediction).iloc[0].values[1],
            'color': postprocess_predictions(prediction).iloc[0].values[2],
            "timestamp": datetime.utcnow(),
        }
        

    return app

def main():
    # Runs app instance using uvicorn and factory pattern.
    uvicorn.run("pipelines.inference_pipeline:create_app", factory=True, port=4200, reload=True)
    # NOTE: url: http://localhost:4200/aqi/online-prediction with this you get the prediction result.

