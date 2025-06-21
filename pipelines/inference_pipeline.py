# NOTE: The inference pipeline is responsible for loading the model and making predictions.
# NOTE: the inference pipeline also contains the following:
# 1. Load Configurations.
# 2. Load Model from `Model Registry`
# 3. Prediction function.
# 4. Return Output to the API.

import os
import sys
import io
import shap
import base64
import matplotlib.pyplot as plt
import uvicorn
import pandas as pd
from pathlib import Path
from typing import Union, List
from pydantic import BaseModel
from datetime import datetime 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Adding the project root.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model_loader import load_model
from src.utils import (config_loader, setup_logger, select_best_model,
                       get_logger, postprocess_predictions, fetch_current_data, read_processed_data)
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
    model_path, acc, roc, weighted_score = select_best_model(os.path.join(project_root, config['model_registry']['model_path'])) 
    if not model_path:
        log.error("No model found in the model registry. Please train a model first.")
        raise HTTPException(status_code=500, detail="No model found in the model registry. Please train a mode first..")

    # Instantiate the model loader.
    model = load_model(model_path=model_path, log=log)   # type: ignore
    log.info(f"\nBest Model metadata:\nNumber of features: {model.n_features_in_}\nBalanced accuracy: {acc}\nROC AUC: {roc}\nWeighted Score: {weighted_score}\nModel Path: {model_path}")
    log.success("Model ready for inference...")

    # Init App instance.
    app  = FastAPI(
        title="AIR QUALITY INDEX SYSTEM.",
        description="A system designed to predict the air quality of any city - Akure."
        )

    @app.get("/")
    def root():
        return {'message': "Welcome to the Air Quality prediction system..."}


    @app.get('/aqi/inference', response_model=PredictionOutput)
    def online_predict():
        # Get the current AQI features :->.
        aqi_features = fetch_current_data(config=config)
        df = pd.DataFrame([aqi_features], index=[0])    # Here: adding index to the dict helps create a 1-row index for the dataframe.
        # Apply feature engineering..
        feature_eng = feature_engineering(df)
        # Drop unnecessary columns.
        feature_eng = feature_eng.drop(columns=['timestamp', 'aqi'], errors='ignore')
        # make prediction using the loaded model..
        prediction = model.predict(feature_eng.values)  
        prediction = prediction + 1  # Adjusting the prediction to match the AQI scale (1-5).

        # json output. 
        return {
            'location': "Akure, Nigeria",
            'predicted_aqi': int(prediction[0]),
            'advice': postprocess_predictions(prediction).iloc[0].values[1],
            'color': postprocess_predictions(prediction).iloc[0].values[2],
            "timestamp": datetime.utcnow(),
        }
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model) 

    @app.get('/aqi/explain', response_class=JSONResponse)
    def explain_prediction():
        """
        Explain the model's AQI prediction for a given input.
        Expects a JSON input with the same features as the model.
        """
        # Get the current AQI features
        data = fetch_current_data(config=config) 
        df = pd.DataFrame([data], index=[0])  # Create a DataFrame with the input data

        # Apply feature engineering
        features_eng = feature_engineering(df)

        # Drop unnecessary columns
        features_eng = features_eng.drop(columns=['timestamp', 'aqi'], errors="ignore")
        # SHAP values 
        shap_values = explainer(features_eng)

        # Generate feature names.
        feature_names = list(features_eng.columns)
        shap_vals = shap_values.values[0].tolist()

        return JSONResponse({
            "features": list(feature_names),
            "shap_values": shap_vals,
            # "expected_value": shap_values.base_values[0].tolist(),
        })




    @app.get('/aqi/plot', response_class=HTMLResponse)
    def test_prediction_plot():
        # Load test data
        test_path = os.path.join(project_root, 'data', 'processed', 'test', 'aqi_test_data_v1.csv')
        test_df = read_processed_data(test_path, log).tail(168)
        #  Ensure timestamp is datetime.
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], errors='coerce')
        # Apply feature engineering
        feature_eng = feature_engineering(test_df)
        X = feature_eng.drop(columns=['timestamp', 'aqi'], errors='ignore')
        y_true = test_df['aqi']
        timestamps = pd.to_datetime(test_df['timestamp'])
        # Predict
        y_pred = model.predict(X.values) + 1  # Adjust scale if needed


        # Plot with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, y_true, label='Actual AQI', color='#23272F')  # DeepSkyBlue
        ax.plot(timestamps, y_pred, label='Predicted AQI', color='#2D3748', linestyle="None", marker='o', markersize=3)  # Tomato

        # Set title and labels with bright colors for readability
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('AQI', color='white')
        ax.set_title('Actual vs Predicted AQI Over 1-WEEK', color='white', fontsize=16)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        fig.patch.set_facecolor('#222')  # Slightly lighter than pure black

        plt.tight_layout()

        # Convert plot to PNG image and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; display:block; margin:auto;"/>'

        return html
        

    return app

def main():
    # Runs app instance using uvicorn and factory pattern.
    uvicorn.run("pipelines.inference_pipeline:create_app", factory=True, host='0.0.0.0', port=10000)
    # NOTE: url: http://localhost:10000/aqi/inference with this you get the prediction result.

if __name__ == "__main__":
    # model_path, _, _, _ = select_best_model(os.path.join(project_root, config['model_registry']['model_path']))
    # model = load_model(model_path=model_path, log=log)   # type: ignore
    # compressed_model = load_model(model_path="C:/Users/user/Desktop/startups/project/aqi_mvp/models/xgb_acc_0.845_roc_0.996.pkl", log=log)   # type: ignore
    # # Load test data
    # test_path = os.path.join(project_root, 'data', 'processed', 'test', 'aqi_test_data_v1.csv')
    # test_df = read_processed_data(test_path, log)
    # #  Ensure timestamp is datetime.
    # test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], errors='coerce')
    # # Apply feature engineering
    # feature_eng = feature_engineering(test_df)
    # X = feature_eng.drop(columns=['timestamp', 'aqi'], errors='ignore')
    # y_true = test_df['aqi']

    # # Test accuracy before and after compression
    # from sklearn import metrics
    # original_accuracy = metrics.balanced_accuracy_score(y_true, model.predict(X) + 1)

    # print(f"Original accuracy: {original_accuracy}")
    pass