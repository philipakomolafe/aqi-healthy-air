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
import numpy as np
from pathlib import Path
from typing import Union, List
from pydantic import BaseModel
from datetime import datetime 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Adding the project root.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model_loader import load_model, auto_detect_and_load_model, get_model_info_from_path
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
        model_info: dict  # Add model information to response


# Helper function to get model features for SHAP
def get_model_features(model, model_type):
    """Get number of features from model based on type."""
    if model_type == 'classical':
        return getattr(model, 'n_features_in_', None)
    elif model_type == 'deep_learning':
        # For deep learning models, we need to check the input shape
        if hasattr(model, 'model') and hasattr(model.model, 'input_shape'):
            # Return the feature dimension (last dimension of input shape)
            input_shape = model.model.input_shape
            if len(input_shape) >= 2:
                return input_shape[-1]  # Feature dimension
        return None
    return None


# Helper function to make predictions
def make_prediction(model, model_type, features, log):
    """Make prediction based on model type."""
    try:
        if model_type == 'classical':
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        elif model_type == 'deep_learning':
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Adjust prediction to match AQI scale (1-5 instead of 0-4)
        prediction = prediction + 1
        
        return prediction, prediction_proba
        
    except Exception as e:
        log.error(f"Error making prediction with {model_type} model: {str(e)}")
        raise


# Helper function to create SHAP explainer
def create_explainer(model, model_type, sample_data, log):
    """Create appropriate SHAP explainer based on model type."""
    try:
        log.info(f"Creating SHAP explainer for {model_type} model...")
        log.info(f"Model type: {type(model)}")
        log.info(f"Sample data shape: {sample_data.shape}")
        
        if model_type == 'classical':
            # Get the actual model class name
            model_name = str(type(model).__name__).lower()
            log.info(f"Classical model detected: {model_name}")
            
            # Try TreeExplainer first for tree-based models
            if any(tree_model in model_name for tree_model in ['randomforest', 'xgb', 'gradient', 'decision', 'forest']):
                log.info("Using TreeExplainer for tree-based model")
                explainer = shap.TreeExplainer(model)
                log.success("TreeExplainer created successfully")
                return explainer
            
            # For other models (SVM, KNN, etc.), use KernelExplainer
            else:
                log.info(f"Using KernelExplainer for {model_name}")
                # Create a wrapper function for prediction
                def predict_fn(X):
                    return model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
                
                explainer = shap.KernelExplainer(predict_fn, sample_data[:50])  # Use smaller sample
                log.success("KernelExplainer created successfully")
                return explainer
                
        elif model_type == 'deep_learning':
            log.info("Creating explainer for deep learning model")
            
            # Create prediction function for deep learning
            def predict_fn(X):
                predictions = model.predict(X)
                # Handle different output shapes
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    return predictions  # Multi-class probabilities
                else:
                    return predictions.flatten()  # Single output
            
            # Use smaller sample for efficiency
            explainer = shap.KernelExplainer(predict_fn, sample_data[:30])
            log.success("KernelExplainer created for deep learning model")
            return explainer
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        log.error(f"Failed to create SHAP explainer: {str(e)}")
        log.error(f"Error type: {type(e).__name__}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return None

# Defining callable to run the Inference Pipeline..
def create_app():
    # Define the model path from the config.
    model_selection_result = select_best_model(os.path.join(project_root, config['model_registry']['model_path'])) 
    
    if len(model_selection_result) == 5:
        model_path, acc, roc, weighted_score, model_type = model_selection_result
    else:
        # Fallback for older version
        model_path, acc, roc, weighted_score = model_selection_result
        model_type = 'classical'  # Default assumption
    
    if not model_path:
        log.error("No model found in the model registry. Please train a model first.")
        raise HTTPException(status_code=500, detail="No model found in the model registry. Please train a model first.")

    # Load the model using auto-detection
    try:
        model, detected_model_type = auto_detect_and_load_model(model_path, log)
        # Use detected type if selection didn't provide it
        if model_type != detected_model_type:
            log.info(f"Model type updated from {model_type} to {detected_model_type} based on auto-detection")
            model_type = detected_model_type
    except Exception as e:
        log.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Get model info
    model_info = get_model_info_from_path(model_path)
    n_features = get_model_features(model, model_type)
    
    log.info(f"\nBest Model metadata:")
    log.info(f"Model Name: {model_info.get('model_name', 'Unknown')}")
    log.info(f"Model Type: {model_type}")
    log.info(f"Format: {model_info.get('format', 'Unknown')}")
    log.info(f"Number of features: {n_features}")
    log.info(f"Balanced accuracy: {acc}")
    log.info(f"ROC AUC: {roc}")
    log.info(f"Weighted Score: {weighted_score}")
    log.info(f"Model Path: {model_path}")
    log.success("Model ready for inference...")

    # Load sample data for SHAP explainer
    explainer = None  # Initialize as None
    try:
        test_path = os.path.join(project_root, 'data', 'processed', 'test', 'aqi_test_data_v1.csv')
        log.info(f"Loading sample data from: {test_path}")
        
        sample_df = read_processed_data(test_path, log).head(200)
        log.info(f"Sample data loaded: {sample_df.shape}")
        
        # Apply feature engineering
        sample_features = feature_engineering(sample_df)
        sample_features = sample_features.drop(columns=['timestamp', 'aqi'], errors='ignore')
        log.info(f"Sample features after engineering: {sample_features.shape}")
        log.info(f"Feature columns: {list(sample_features.columns)}")
        
        # Create SHAP explainer
        explainer = create_explainer(model, model_type, sample_features.values, log)
        
        if explainer is not None:
            log.success("SHAP explainer created successfully")
        else:
            log.warning("SHAP explainer creation failed - explanations will not be available")
            
    except Exception as e:
        log.error(f"Could not initialize SHAP explainer: {str(e)}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        explainer = None

    # Init App instance.
    app = FastAPI(
        title="AIR QUALITY INDEX SYSTEM.",
        description="A system designed to predict the air quality of any city - Akure."
    )

    @app.get("/")
    def root():
        return {
            'message': "Welcome to the Air Quality prediction system...",
            'model_info': {
                'model_name': model_info.get('model_name', 'Unknown'),
                'model_type': model_type,
                'accuracy': acc,
                'roc_auc': roc
            }
        }

    @app.get('/aqi/inference', response_model=PredictionOutput)
    def online_predict():
        try:
            # Get the current AQI features
            aqi_features = fetch_current_data(config=config)
            df = pd.DataFrame([aqi_features], index=[0])
            
            # Apply feature engineering
            feature_eng = feature_engineering(df)
            
            # Drop unnecessary columns
            features = feature_eng.drop(columns=['timestamp', 'aqi'], errors='ignore')
            
            # Make prediction using the loaded model
            prediction, prediction_proba = make_prediction(model, model_type, features.values, log)

            # Postprocess predictions
            postprocessed = postprocess_predictions(prediction)
            
            return {
                'location': "Akure, Nigeria",
                'predicted_aqi': int(prediction[0]),
                'advice': postprocessed.iloc[0]['aqi_category'],
                'color': postprocessed.iloc[0]['aqi_health_color'],
                "timestamp": datetime.utcnow(),
                'model_info': {
                    'model_name': model_info.get('model_name', 'Unknown'),
                    'model_type': model_type,
                    'confidence': float(np.max(prediction_proba[0])) if prediction_proba is not None else None
                }
            }
        except Exception as e:
            log.error(f"Error in online prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get('/aqi/explain', response_class=JSONResponse)
    def explain_prediction():
        """
        Explain the model's AQI prediction for a given input.
        """
        if explainer is None:
            error_msg = f"SHAP explainer not available for {model_type} model. This could be due to model compatibility issues or initialization failures."
            log.warning(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
        
        try:
            log.info("Starting SHAP explanation...")
            
            # Get the current AQI features
            data = fetch_current_data(config=config) 
            df = pd.DataFrame([data], index=[0])
            log.info(f"Input data shape: {df.shape}")

            # Apply feature engineering
            features_eng = feature_engineering(df)
            features_eng = features_eng.drop(columns=['timestamp', 'aqi'], errors="ignore")
            log.info(f"Engineered features shape: {features_eng.shape}")
            log.info(f"Feature columns: {list(features_eng.columns)}")
            
            # Calculate SHAP values based on explainer type
            log.info(f"Explainer type: {type(explainer)}")
            
            if hasattr(explainer, 'shap_values'):
                # For TreeExplainer and some others
                log.info("Using explainer.shap_values() method")
                shap_values = explainer.shap_values(features_eng.values)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first class for multi-class
            else:
                # For KernelExplainer and others
                log.info("Using explainer() call method")
                shap_result = explainer(features_eng.values)
                if hasattr(shap_result, 'values'):
                    shap_values = shap_result.values[0]
                else:
                    shap_values = shap_result

            # Process SHAP values
            feature_names = list(features_eng.columns)
            
            if isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) > 1:
                    shap_vals = shap_values[0].tolist() if shap_values.shape[0] == 1 else shap_values.flatten().tolist()
                else:
                    shap_vals = shap_values.tolist()
            else:
                shap_vals = [float(shap_values)] if np.isscalar(shap_values) else list(shap_values)

            log.info(f"SHAP values calculated: {len(shap_vals)} values for {len(feature_names)} features")

            return JSONResponse({
                "model_type": model_type,
                "model_name": model_info.get('model_name', 'Unknown'),
                "features": feature_names,
                "shap_values": shap_vals,
                "feature_values": features_eng.iloc[0].tolist(),
                "explanation_method": type(explainer).__name__
            })
            
        except Exception as e:
            log.error(f"Error in SHAP explanation: {str(e)}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

    @app.get('/aqi/plot', response_class=HTMLResponse)
    def test_prediction_plot():
        try:
            # Load test data
            test_path = os.path.join(project_root, 'data', 'processed', 'test', 'aqi_test_data_v1.csv')
            test_df = read_processed_data(test_path, log).tail(168)
            
            # Ensure timestamp is datetime
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], errors='coerce')
            
            # Apply feature engineering
            feature_eng = feature_engineering(test_df)
            X = feature_eng.drop(columns=['timestamp', 'aqi'], errors='ignore')
            y_true = test_df['aqi']
            timestamps = pd.to_datetime(test_df['timestamp'])
            
            # Make predictions
            y_pred, _ = make_prediction(model, model_type, X.values, log)

            # Plot with dark theme
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, y_true, label='Actual AQI', color='#00FF7F', linewidth=2)
            ax.plot(timestamps, y_pred, label=f'Predicted AQI ({model_type})', color="#EEF0F4", 
                   linestyle="None", marker='o', markersize=4)

            # Set title and labels
            ax.set_xlabel('Time', color='white')
            ax.set_ylabel('AQI', color='white')
            ax.set_title(f'Actual vs Predicted AQI Over 1-WEEK ({model_info.get("model_name", "Unknown")} Model)', 
                        color='white', fontsize=16)
            ax.tick_params(colors='white')
            ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
            fig.patch.set_facecolor('#222')

            plt.tight_layout()

            # Convert plot to PNG and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; display:block; margin:auto;"/>'

            return html
            
        except Exception as e:
            log.error(f"Error generating plot: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")

    return app

def main():
    # Runs app instance using uvicorn and factory pattern.
    uvicorn.run("pipelines.inference_pipeline:create_app", factory=True, host='0.0.0.0', port=10000)

if __name__ == "__main__":
    pass
