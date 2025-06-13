# Air Quality Index (AQI) Machine Learning Project

This project is a modular, production-ready framework for analyzing, modeling, and predicting Air Quality Index (AQI) using machine learning. It is designed for reproducibility, extensibility, and clarity, with each component separated for ease of understanding and modification.

## Project Overview

The project is organized into several key components:

- **Data Ingestion & Cleaning**: Fetches raw AQI data, cleans it, and prepares it for analysis.
- **Feature Engineering & Selection**: Extracts meaningful features and selects the most relevant ones.
- **Model Training & Evaluation**: Trains multiple models, evaluates their performance, and saves the best ones.
- **Inference Pipeline**: Loads trained models and serves predictions via an API.
- **Configuration & Logging**: Centralized configuration and logging for reproducibility.

## Directory Structure & Key Files

```
aqi_mvp/
│
├── config/
│   └── config.yaml           # Central configuration for data paths, API keys, etc.
│
├── data/
│   ├── raw/                  # Raw AQI data (e.g., aqi_data_2016-2025.csv)
│   └── processed/
│       └── train/val/test/   # Cleaned and split datasets for ML
│
├── logs/
│   └── pipeline.log          # Logs for pipeline runs
│
├── models/                   # Saved model artifacts (e.g., .pkl files)
│
├── pipelines/
│   ├── feature_pipeline.py   # Orchestrates data cleaning, feature engineering, and selection
│   ├── train_pipeline.py     # Handles model training and evaluation
│   └── inference_pipeline.py # Loads models and serves predictions (API-ready)
│
├── src/
│   ├── data_cleaner.py       # Functions for loading and cleaning raw data
│   ├── data_fetcher.py       # (Optional) For fetching data from APIs
│   ├── data_splitter.py      # Splits data into train/val/test sets
│   ├── feature_engineering.py# Adds new features to the dataset
│   ├── feature_selection.py  # Selects important features
│   ├── feature_storage.py    # Handles saving/versioning of features
│   ├── model_loader.py       # Loads models for inference
│   ├── neptune_utils.py      # Utilities for experiment tracking (Neptune.ai)
│   ├── train.py              # Core model training logic
│   └── utils.py              # Config loader, logger, and helpers
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## How Each Component Works

### 1. Configuration (`config/config.yaml`)
All paths, API keys, and settings are centralized here. Change this file to point to your own data or adjust parameters.

### 2. Data Handling
- **Raw Data**: Place your source CSVs in `data/raw/`.
- **`src/data_cleaner.py`**: Loads and cleans raw data, handling missing values and formatting.
- **`src/data_splitter.py`**: Splits cleaned data into train/val/test sets for robust model evaluation.

### 3. Feature Engineering & Selection
- **`src/feature_engineering.py`**: Adds time-based, interaction, and domain-specific features (e.g., rolling AQI stats, ratios).
- **`src/feature_selection.py`**: Uses correlation and importance metrics to select the best features.

### 4. Pipelines
- **`pipelines/feature_pipeline.py`**: Orchestrates the full feature pipeline—loading, cleaning, engineering, selecting, and saving features.
- **`pipelines/train_pipeline.py`**: Loads processed data, trains models (KNN, RF, SVC, XGB), evaluates them, and logs results.
- **`pipelines/inference_pipeline.py`**: Loads the best model and exposes a FastAPI endpoint for predictions.

### 5. Model Management
- **`models/`**: Stores all trained model artifacts, named with their accuracy and ROC for easy selection.
- **`src/model_loader.py`**: Loads models for inference.

### 6. Experiment Tracking
- **`src/neptune_utils.py`**: Integrates with Neptune.ai for experiment tracking and artifact logging.

### 7. Utilities
- **`src/utils.py`**: Handles configuration loading, logger setup, and other helper functions.

### 8. Logging
- **`logs/pipeline.log`**: All pipeline runs and errors are logged here for debugging and reproducibility.

## Reproducibility Steps

1. **Clone the repository**  
   `git clone <repo-url> && cd aqi_mvp`

2. **Install dependencies**  
   `pip install -r requirements.txt`

3. **Configure your environment**  
   Edit `config/config.yaml` as needed.

4. **Prepare data**  
   Place your raw AQI data in `data/raw/`.

5. **Run the feature pipeline**  
   `python pipelines/feature_pipeline.py`

6. **Run the training pipeline**  
   `python pipelines/train_pipeline.py`

7. **Run the inference pipeline (API)**  
   `python pipelines/inference_pipeline.py`

## Notes

- All scripts are modular and can be run independently.
- The project is designed for easy extension—add new models, features, or data sources as needed.
- For experiment tracking, set up Neptune.ai and update your config.
