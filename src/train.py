# `train.py` contains the functions to train the models based off the features and labels 
# In the saved train, val (validation) and test files within the processed folder. 
# Gotten from `feature_storage.py`.

import numpy as np
from time import sleep
import pandas as pd
import tempfile
import zipfile
import shutil
import joblib
import os
from pathlib import Path
from typing import Tuple, Union 
from sklearn.base import BaseEstimator
import xgboost
from sklearn import metrics, svm, ensemble, neighbors
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from .neptune_utils import init_neptune_run, log_model, log_metrics, stop_run
# from .utils import config_loader, setup_logger, get_logger, read_processed_data


def create_sequences(X, y, sequence_length=24):
    """
    Create sequences for time series deep learning models.
    
    Args:
        X: Feature array
        y: Target array
        sequence_length: Number of time steps to look back
    
    Returns:
        X_seq: Sequences of features
        y_seq: Corresponding targets
    """
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def create_gru_model(input_shape, num_classes, units=64, dropout_rate=0.3):
    """Create GRU model for sequence classification."""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        GRU(units//2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units//4, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_lstm_model(input_shape, num_classes, units=64, dropout_rate=0.3):
    """Create LSTM model for sequence classification."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(units//2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units//4, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_hybrid_model(input_shape, num_classes, gru_units=32, lstm_units=32, dropout_rate=0.3):
    """Create hybrid GRU-LSTM model."""
    model = Sequential([
        GRU(gru_units, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(32, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class DeepLearningWrapper:
    """Wrapper to make Keras models compatible with sklearn metrics."""
    
    def __init__(self, model, scaler=None, sequence_length=24, neptune_run=None):
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.classes_ = None
        self.model_name = None
        self.neptune_run = neptune_run  # Add Neptune run for logging
        self.training_history = None  # Store training history

    def fit(self, X, y):
        # Scale features if scaler provided
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y, self.sequence_length)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences with length {self.sequence_length}")
        
        # Store unique classes
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        
        # Convert to categorical
        y_categorical = to_categorical(y_seq, num_classes=num_classes)
        
        # Define callbacks
        early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.65, min_lr=1e-6)
        
        # Add Neptune callback if run is provided
        callbacks = [early_stopping, reduce_lr]
        
        if self.neptune_run:
            # Create custom Neptune callback
            class NeptuneCallback(tf.keras.callbacks.Callback):
                def __init__(self, run, model_name):
                    super().__init__()
                    self.run = run
                    self.model_name = model_name
                
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        # Log training metrics for each epoch
                        for metric, value in logs.items():
                            self.run[f"{self.model_name}/train/{metric}"].log(value)
            
            callbacks.append(NeptuneCallback(self.neptune_run, self.model_name))
        
        # Train model
        history = self.model.fit(
            X_seq, y_categorical,
            epochs=100,
            batch_size=32,
            validation_split=0.3,
            callbacks=callbacks,
            verbose=1   
        )
        
        # Store training history
        self.training_history = history.history
        
        # Log final training summary to Neptune
        if self.neptune_run and self.training_history:
            self._log_training_summary()
        
        return self
    
    def _log_training_summary(self):
        """Log training summary metrics to Neptune."""
        if not self.training_history or not self.neptune_run:
            return
        
        # Log final metrics
        final_metrics = {
            f"{self.model_name}/final/train_loss": self.training_history['loss'][-1],
            f"{self.model_name}/final/train_accuracy": self.training_history['accuracy'][-1],
            f"{self.model_name}/final/val_loss": self.training_history['val_loss'][-1],
            f"{self.model_name}/final/val_accuracy": self.training_history['val_accuracy'][-1],
            f"{self.model_name}/epochs_trained": len(self.training_history['loss']),
            f"{self.model_name}/best_val_loss": min(self.training_history['val_loss']),
            f"{self.model_name}/best_val_accuracy": max(self.training_history['val_accuracy'])
        }
        
        for metric, value in final_metrics.items():
            self.neptune_run[metric] = value
    
    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a deep learning model from directory.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            DeepLearningWrapper instance with loaded model
        """
        try:
            model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
            # Initialize variables
            keras_model = None
            
            # Look for the keras model file
            keras_model_path = model_path / 'keras_model.h5'
            
            if keras_model_path.exists():
                # Load the h5 model
                keras_model = tf.keras.models.load_model(str(keras_model_path))
            else:
                # Try alternative names
                for alt_name in ['model.h5', 'model.keras']:
                    alt_path = model_path / alt_name
                    if alt_path.exists():
                        keras_model = tf.keras.models.load_model(str(alt_path))
                        break
                
                # If no h5/keras files found, try SavedModel format
                if keras_model is None and (model_path / 'saved_model.pb').exists():
                    keras_model = tf.keras.models.load_model(str(model_path))
                
                # If still no model found, raise error
                if keras_model is None:
                    raise FileNotFoundError(f"No Keras model found in {model_path}")
            
            # Load metadata if available
            metadata_path = model_path / 'metadata.pkl'
            scaler = None
            sequence_length = 24
            model_name = model_path.name
            
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = joblib.load(f)
                
                scaler = metadata.get('scaler')
                sequence_length = metadata.get('sequence_length', 24)
                model_name = metadata.get('model_name', model_path.name)
            
            # Create wrapper instance
            wrapper = cls(keras_model, scaler, sequence_length)
            wrapper.model_name = model_name
            
            # Set classes if available in metadata
            if metadata_path.exists() and 'classes_' in metadata:
                wrapper.classes_ = metadata['classes_']
            
            return wrapper
            
        except Exception as e:
            raise Exception(f"Error loading deep learning model from {model_path}: {str(e)}")


    @classmethod
    def load_from_zip(cls, zip_path: str):
        """
        Load a deep learning model from zip file.
        
        Args:
            zip_path: Path to zip file
            
        Returns:
            DeepLearningWrapper instance with loaded model
        """

        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                
                # Load from extracted directory
                wrapper = cls.load_model(temp_dir)
                return wrapper
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            raise Exception(f"Error loading deep learning model from zip {zip_path}: {str(e)}")
    
    def save_model(self, save_dir: str, model_name: str, accuracy: float, roc_auc:float):
        """
        Save the deep learning model to directory and create zip.
        
        Args:
            save_dir: Directory to save model
            model_name: Name of the model
            accuracy: Model accuracy
            roc_auc: Model ROC AUC score
            
        Returns:
            Tuple of (model_directory_path, zip_file_path)
        """
        try:
            # Create model directory
            model_dir_name = f"{model_name}_acc_{accuracy:.3f}_roc_{roc_auc:.3f}"
            model_dir_path = os.path.join(save_dir, model_dir_name)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # Save Keras model
            keras_model_path = os.path.join(model_dir_path, 'keras_model.h5')
            self.model.save(keras_model_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'classes_': self.classes_,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'training_history': self.training_history
            }
            
            metadata_path = os.path.join(model_dir_path, 'metadata.pkl')
            with open(metadata_path, 'wb') as f:
                joblib.dump(metadata, f)
            
            # Create zip file
            zip_path = f"{model_dir_path}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(model_dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, save_dir)
                        zipf.write(file_path, arcname)
            
            return model_dir_path, zip_path
     
        except Exception as e:
            raise Exception(f"Error saving deep learning model: {str(e)}")

    
    def predict(self, X):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Ensure input is numpy array before parsing.
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale input features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Create sequences for prediction
        if len(X_scaled) >= self.sequence_length:
            X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
            if len(X_seq) > 0:
                predictions = self.model.predict(X_seq)
                
                # Handle different output formats
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class output - get class predictions
                    predicted_classes = np.argmax(predictions, axis=1)
                else:
                    # Single output - convert to classes
                    predicted_classes = self._convert_to_classes(predictions.flatten())
                
                return predicted_classes
            else:
                # Not enough data for sequences, return dummy prediction
                return np.array([2])  # Default to class 2 (moderate)
        else:
            # Not enough data for sequences
            return np.array([2] * len(X_scaled))


    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure input is numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Create sequences for prediction
        if len(X_scaled) >= self.sequence_length:
            X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
            if len(X_seq) > 0:
                predictions = self.model.predict(X_seq)
                
                # Handle different output formats
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class probabilities
                    return predictions
                else:
                    # Single output - create dummy probabilities
                    predicted_classes = self._convert_to_classes(predictions.flatten())
                    prob_matrix = np.zeros((len(predicted_classes), 5))
                    for i, cls in enumerate(predicted_classes):
                        prob_matrix[i, cls] = 1.0
                    return prob_matrix
            else:
                # Not enough data, return dummy probabilities
                prob_matrix = np.zeros((1, 5))
                prob_matrix[0, 1] = 1.0  # Default to class 2 (moderate)
                return prob_matrix
        else:
            # Not enough data for sequences
            prob_matrix = np.zeros((len(X_scaled), 5))
            prob_matrix[:, 1] = 1.0  # Default to class 2 (moderate)
            return prob_matrix

        
    def _convert_to_classes(self, predictions):
        """Convert continuous predictions to AQI classes (1-5)"""
        # Define the AQI breakpoints for our use case.
        bins = [0, 50, 100, 150, 200, float('inf')]
        classes = np.digitize(predictions, bins)
        return np.clip(classes, 1, 5)







# NOTE: Define the model training function. 
def train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, log, config):
    """
    Train the model using the training and validation data.
    """
    log.info("Training model...")

    # Define params for hyperparameter tuning.
    svc_params = {
        'C': np.logspace(-2, 2, 10),  # log scale
        'kernel': ['rbf'],
        'gamma': np.logspace(-4, 0, 6).tolist() + ['scale', 'auto'],  # 0.0001 to 1, plus 'scale'/'auto'
        'class_weight': [None, 'balanced']
    }

    knn_params = {
        'n_neighbors': list(range(1, 21)),  # 1 to 20 for smooth plots
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    rf_params = {
        'n_estimators': list(range(50, 501, 10)),  # 50 to 500 in steps of 10
        'max_depth': [None, 5, 10, 20, 50]
    }

    xgb_params = {
        'n_estimators': list(range(50, 501, 10)),  # 50 to 500 in steps of 10
        'learning_rate': np.logspace(-3, 0, 8),   # 0.001 to 1, log scale
        'max_depth': [3, 5, 7, 10]
    }

    # Initialize Neptune run
    run = init_neptune_run(
        project_name=config.get('neptune', {}).get('project_name'),
        api_token=config.get('neptune', {}).get('api_token'),
        tags=["train", "multi-model", "deep-learning"],
        params={
            'svc_params': str(svc_params),
            'knn_params': str(knn_params),
            'rf_params': str(rf_params),
            'xgb_params': str(xgb_params),
            "validation_split": 0.3,
            'cv_splits': 3
        }
    )

    # XGBoost only reads from [0 - n] not [1 - n]
    # where n -> nth occurence.
    train_data['aqi'] = train_data['aqi'] - 1
    val_data['aqi'] = val_data['aqi'] - 1 

    # Preserve the timestamps for later use. 
    train_timestamps = train_data['timestamp']
    val_timestamps = val_data['timestamp']
    
    # Drop 'timestamp' column if present (XGBoost can't handle object types)
    drop_cols = ['aqi']
    if 'timestamp' in train_data.columns:
        drop_cols.append('timestamp')

    # Seperate into features and target.
    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data['aqi']
    X_val = val_data.drop(columns=drop_cols)
    y_val = val_data['aqi']

    # Adjust Target to represent "categorized data"
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Get number of classes and features for deep learning models
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    num_features = X_train.shape[1]
    sequence_length = min(24, len(X_train) // 4)  # Adjust based on data size

    # Define model instances.
    # Classical ML models
    svc = lambda: svm.SVC(probability=True)
    knn = lambda: neighbors.KNeighborsClassifier()
    rf = lambda: ensemble.RandomForestClassifier()
    xgb = lambda: xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Deep Learning models - Pass Neptune run for logging
    def create_gru_wrapper():
        scaler = StandardScaler()
        model = create_gru_model((sequence_length, num_features), num_classes)
        wrapper = DeepLearningWrapper(model, scaler, sequence_length, neptune_run=run)
        wrapper.model_name = 'gru'  # Set model name
        return wrapper

    def create_lstm_wrapper():
        scaler = StandardScaler()
        model = create_lstm_model((sequence_length, num_features), num_classes)
        wrapper = DeepLearningWrapper(model, scaler, sequence_length, neptune_run=run)
        wrapper.model_name = 'lstm'  # Set model name
        return wrapper

    def create_hybrid_wrapper():
        scaler = StandardScaler()
        model = create_hybrid_model((sequence_length, num_features), num_classes)
        wrapper = DeepLearningWrapper(model, scaler, sequence_length, neptune_run=run)
        wrapper.model_name = 'hybrid'  # Set model name
        return wrapper

    # Define time-based cv split.
    tscv = model_selection.TimeSeriesSplit(n_splits=3)  # Reduced for deep learning

    # Define the scoring metrics.
    scoring = {
        'accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),
        'f1': metrics.make_scorer(metrics.f1_score, average='macro', zero_division=0),
        'recall': metrics.make_scorer(metrics.recall_score, average='macro', zero_division=0),
        'precision': metrics.make_scorer(metrics.precision_score, average='macro', zero_division=0),
        'roc_auc': 'roc_auc_ovr',
    }
    
    # Log validation strategy
    log.info(f"Using TimeSeriesSplit with {tscv.n_splits} splits")
    log.info(f"Deep learning models use {0.3*100:.0f}% validation split")

    # Define RandomizedSearchCV for hyperparameter tuning.
    log.info("Training Classical ML models...")
    svc_search = model_selection.RandomizedSearchCV(svc(), svc_params, scoring=scoring, refit="f1", n_iter=10, cv=tscv, n_jobs=-1, verbose=2)
    knn_search = model_selection.RandomizedSearchCV(knn(), knn_params, scoring=scoring, refit='f1', n_iter=10, cv=tscv, n_jobs=-1, verbose=2)
    rf_search = model_selection.RandomizedSearchCV(rf(), rf_params, scoring=scoring, refit='f1', n_iter=11, cv=tscv, n_jobs=-1, verbose=2)
    xgb_search = model_selection.RandomizedSearchCV(xgb(), xgb_params, scoring=scoring, refit="f1", n_iter=12, cv=tscv, n_jobs=-1, verbose=2)
    
    # Fit classical ML models
    log.info(f"\nFitted Features: {X_train.columns}\nTarget used: {y_train.name}")
    log.info("Fitting classical ML models...")
    
    classical_models = []
    for name, search in zip(['svc', 'knn', 'rf', 'xgb'], [svc_search, knn_search, rf_search, xgb_search]):
        log.info(f'Fitting {name.upper()} model...')
        search.fit(X_train, y_train)
        classical_models.append(search.best_estimator_)
        
        # Log classical ML CV results to Neptune
        cv_results = search.cv_results_
        best_score = search.best_score_
        run[f"{name}/best_cv_f1_score"] = best_score
        run[f"{name}/best_params"] = str(search.best_params_)
        
        sleep(5)

    # Train Deep Learning models
    log.info("Training Deep Learning models with epoch-by-epoch logging...")
    dl_models = []
    dl_names = ['gru', 'lstm', 'hybrid']
    
    for name, create_model in zip(dl_names, [create_gru_wrapper, create_lstm_wrapper, create_hybrid_wrapper]):
        log.info(f'Training {name.upper()} model...')
        try:
            model = create_model()
            # The fit method will now log epoch-by-epoch metrics automatically
            model.fit(X_train.values, y_train.values)
            dl_models.append(model)
        except Exception as e:
            log.warning(f"Failed to train {name} model: {str(e)}")
            dl_models.append(None)
        sleep(10)

    # Combine all models
    all_models = classical_models + [m for m in dl_models if m is not None]
    all_model_names = ['svc', 'knn', 'rf', 'xgb'] + [name for name, model in zip(dl_names, dl_models) if model is not None]

    # Create best_models dictionary
    best_models = {}
    for name, model in zip(all_model_names, all_models):
        best_models[name] = model

    # Store CV results for classical models
    results = {
        'svc': svc_search.cv_results_,
        'knn': knn_search.cv_results_,
        'rf': rf_search.cv_results_,
        'xgb': xgb_search.cv_results_,
    }

    # Save CV results for classical models
    for model_name, result in results.items():
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['cv_result'])
        file = f"{model_name}_results.csv"
        df_path = os.path.join(path, file)
        df_result = pd.DataFrame(result)
        df_result.to_csv(df_path, index=False)

    # Evaluate all models on validation set
    val_model_performance = {}

    for model_name, model in best_models.items():
        log.info(f"Evaluating {model_name} on validation set...")
        try:
            # Adjust validation data for deep learning models
            if model_name in ['gru', 'lstm', 'hybrid']:
                # For deep learning models, we need to account for sequence length
                min_samples = model.sequence_length
                if len(X_val) < min_samples:
                    log.warning(f"Not enough validation samples for {model_name}. Skipping.")
                    continue
                
                y_pred = model.predict(X_val.values)
                y_pred_proba = model.predict_proba(X_val.values)
                
                # Adjust y_val to match prediction length
                y_val_adjusted = y_val.values[min_samples:]
            else:
                # Classical ML models
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
                y_val_adjusted = y_val.values

            # Calculate metrics
            if len(y_pred) > 0 and len(y_val_adjusted) > 0:
                accuracy = metrics.balanced_accuracy_score(y_val_adjusted, y_pred)
                f1 = metrics.f1_score(y_val_adjusted, y_pred, average='macro', zero_division=0)
                recall = metrics.recall_score(y_val_adjusted, y_pred, average='macro', zero_division=0)
                precision = metrics.precision_score(y_val_adjusted, y_pred, average='macro', zero_division=0)
                
                # Handle ROC AUC calculation
                try:
                    roc_auc = metrics.roc_auc_score(y_val_adjusted, y_pred_proba, multi_class='ovr')
                except Exception as e:
                    log.warning(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
                    roc_auc = 0.0

                # Store performance
                val_model_performance[model_name] = (model, accuracy, f1, recall, precision, roc_auc)

                # Log the metrics
                log.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, ROC AUC: {roc_auc:.3f}")

                # Log final validation metrics to Neptune
                log_metrics(run, {
                    f"{model_name}/validation/accuracy": float(accuracy),
                    f"{model_name}/validation/f1": float(f1),
                    f"{model_name}/validation/recall": float(recall),
                    f"{model_name}/validation/precision": float(precision),
                    f"{model_name}/validation/roc_auc": float(roc_auc)
                })

                # Save the models
                if model_name in ['gru', 'lstm', 'hybrid']:
                    # Save deep learning model in proper format
                    model_path, zip_path = model.save_model(
                        os.path.join(os.path.dirname(os.path.dirname(__file__)), config['model_registry']['model_path']), 
                        model_name, 
                        accuracy, 
                        roc_auc
                    )
                    log.info(f"Deep learning model saved to: {model_path}")
                    log.info(f"Deep learning model zip created: {zip_path}")
                    
                    # Upload zipped model to Neptune (Neptune can handle zip files)
                    try:
                        log_model(run, zip_path, alias=f"{model_name}_model")
                        log.info(f"Deep learning model {model_name} logged to Neptune successfully")
                    except Exception as e:
                        log.warning(f"Failed to log {model_name} model to Neptune: {str(e)}")
                        log.warning(f"Model files are located at: {zip_path}")
                else:
                    # Save classical ML model as pickle
                    model_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), 
                        config['model_registry']['model_path'], 
                        f"{model_name}_acc_{accuracy:.3f}_roc_{roc_auc:.3f}.pkl"
                    )
                    
                    # For classical models, remove unnecessary attributes
                    # This help reduce model size or weight (in MB) so it could be deployment compartible.
                    for attr in ['X_train_', 'y_train_', "oob_score_", 'oob_decision_function']:
                        if hasattr(model, attr):
                            delattr(model, attr)

                    # Save model
                    joblib.dump(model, model_path)
                    log.info(f"Classical ML model saved to: {model_path}")
                    # Upload model to Neptune
                    log_model(run, model_path, alias=f"{model_name}_model")

        except Exception as e:
            log.error(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Find best model based on F1 score
    if val_model_performance:
        best_model_name = max(val_model_performance, key=lambda name: val_model_performance[name][2])  # F1 score is at index 2
        best_val_model = val_model_performance[best_model_name][0]

        # Create validation results DataFrame
        records = []
        for name, (model, accuracy, f1, recall, precision, roc_auc) in val_model_performance.items():
            records.append({
                'model': name,
                'model_type': 'Deep Learning' if name in ['gru', 'lstm', 'hybrid'] else 'Classical ML',
                'accuracy': accuracy,
                'f1': f1,
                'recall': recall,
                'precision': precision,
                'roc_auc': roc_auc
            })

        val_df_results = pd.DataFrame(records)
        val_df_results = val_df_results.sort_values('f1', ascending=False)

        # Save validation results
        val_results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['cv_result'], "val_model_performance.csv")
        val_df_results.to_csv(val_results_path, index=False)
        
        log.info(f"Best model based on F1 score: {best_model_name} with F1: {val_model_performance[best_model_name][2]:.3f}")
        log.info(f"Model rankings saved to: {val_results_path}")
        
        # Stop Neptune run
        stop_run(run)

        return best_val_model
    else:
        log.error("No models were successfully trained and evaluated!")
        stop_run(run)
        return None


def retrain_model(train_data: pd.DataFrame, val_data: pd.DataFrame, log, model, config):
    """Retrain the best model on combined train and validation data."""
    log.info("Retraining best model on combined train+val data...")
    
    # Combine datasets
    X_train_val = pd.concat([train_data, val_data])
    X_train_val = X_train_val.sort_values('timestamp').reset_index(drop=True)
    y_train_val = X_train_val['aqi']
    X_train_val = X_train_val.drop(['aqi', 'timestamp'], axis=1)

    # Check if it's a deep learning model
    if isinstance(model, DeepLearningWrapper):
        final_model = model.fit(X_train_val.values, y_train_val.values)
    else:
        final_model = model.fit(X_train_val, y_train_val)
    
    # Save final model
    model_registry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['model_registry']['model_path'])
    os.makedirs(model_registry_path, exist_ok=True)
    final_model_path = os.path.join(model_registry_path, "best_model.pkl")
    
    joblib.dump(final_model, final_model_path)
    log.info(f"Final retrained model saved to: {final_model_path}")

    return final_model


if __name__ == "__main__":
    pass
