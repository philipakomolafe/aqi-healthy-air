# `train.py` contains the functions to train the models based off the features and labels 
# In the saved train, val (validation) and test files within the processed folder. 
# Gotten from `feature_storage.py`.

import numpy as np
from time import sleep
import pandas as pd
import joblib
import os
from typing import Tuple, Union 
from sklearn.base import BaseEstimator
import xgboost
from sklearn import metrics, svm, ensemble, neighbors
from sklearn import model_selection
from .neptune_utils import init_neptune_run, log_model, log_metrics, stop_run
# from .utils import config_loader, setup_logger, get_logger, read_processed_data




def train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, log, config):
    """
    Train the model using the training and validation data.
    """
    log.info("Training model...")

    # Define params for hyperparameter tuning.
    svc_params = {
        'C': np.logspace(-2, 2, 10),  # 0.01 to 100, log scale
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
        tags=["train", "multi-model"],
        params={
            'svc_params': str(svc_params),
            'knn_params': str(knn_params),
            'rf_params': str(rf_params),
            'xgb_params': str(xgb_params)
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


    # Define model instances.
    # why lambda?  -> Helps create new instance of the model so prevent it from eeing the datA..
    svc = lambda: svm.SVC(probability=True)
    knn = lambda: neighbors.KNeighborsClassifier()
    rf = lambda: ensemble.RandomForestClassifier()
    xgb = lambda: xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    
    # Define the hyperparameter search algorithm.
    # Define time-based cv split.
    tscv = model_selection.TimeSeriesSplit(n_splits=5)

    # Define the scoring metrics.
    scoring = {'accuracy': metrics.make_scorer(metrics.balanced_accuracy_score),
                'f1': metrics.make_scorer(metrics.f1_score, average='macro', zero_division=0),
                'recall': metrics.make_scorer(metrics.recall_score, average='macro', zero_division=0),
                'precision': metrics.make_scorer(metrics.precision_score, average='macro', zero_division=0),
                'roc_auc': 'roc_auc_ovr',
                }

    # Define RandomizedSearchCV for hyperparameter tuning.
    # For selecting the number of iterations (n_iter) - we selected 84% of the sample space..
    svc_search = model_selection.RandomizedSearchCV(svc(), svc_params, scoring=scoring, refit="f1", n_iter=10, cv=tscv, n_jobs=-1, verbose=2)
    sleep(10)
    knn_search = model_selection.RandomizedSearchCV(knn(), knn_params, scoring=scoring, refit='f1', n_iter=10, cv=tscv, n_jobs=-1, verbose=2)
    sleep(10)
    rf_search = model_selection.RandomizedSearchCV(rf(), rf_params, scoring=scoring, refit='f1', n_iter=11, cv=tscv, n_jobs=-1, verbose=2)
    sleep(10)
    xgb_search = model_selection.RandomizedSearchCV(xgb(), xgb_params, scoring=scoring, refit="f1", n_iter=12, cv=tscv, n_jobs=-1, verbose=2)
    sleep(10)
    
    # Fit the models.
    log.info(f"\nFitted Features: {X_train.columns}\nTarget used: {y_train.name}")
    log.info("Fitting models...")
    svc_search.fit(X_train, y_train)
    knn_search.fit(X_train, y_train)
    rf_search.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)

    # Get the best models.
    svc_best = svc_search.best_estimator_
    knn_best = knn_search.best_estimator_   
    rf_best = rf_search.best_estimator_
    xgb_best = xgb_search.best_estimator_

    best_models = {'svc': svc_best,
                    'knn': knn_best,
                    'rf': rf_best,
                    'xgb': xgb_best,
                    }

    # store metric performance on overall models.
    results = {
        'svc': svc_search.cv_results_,
        'knn': knn_search.cv_results_,
        'rf': rf_search.cv_results_,
        'xgb': xgb_search.cv_results_,
    }


    # Log the best models and their parameters.
    for model_name, result in results.items():
        # Define the path to save results.
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['cv_result'])
        file = f"{model_name}_results.csv"
        df_path = os.path.join(path, file)
        # Save to Data
        df_result = pd.DataFrame(result)
        df_result.to_csv(df_path, index=False)

    sleep(10)

    # Define the dict for validation metrics storage.
    val_model_performance = {}

    for model_name, model in best_models.items():
        # Get the predictions.
        y_pred = model.predict(X_val)               # type: ignore
        y_pred_proba = model.predict_proba(X_val)   # type: ignore

        # Get the metrics.
        accuracy = metrics.balanced_accuracy_score(y_val, y_pred)
        f1 = metrics.f1_score(y_val, y_pred, average='macro')
        recall = metrics.recall_score(y_val, y_pred, average='macro')
        precision = metrics.precision_score(y_val, y_pred, average='macro')
        roc_auc = metrics.roc_auc_score(y_val, y_pred_proba, multi_class='ovr')

        # Store balanced-accuracy to `val_accuracy_evals` to be used for final model retraining.
        val_model_performance[model_name] = (model, accuracy, f1, recall, precision, roc_auc)

        # Log the metrics.
        log.info("Logging the model perfomance on Validation data...")
        log.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, ROC AUC: {roc_auc:.3f}")

        # Log metrics to Neptune
        log_metrics(run, {
            f"{model_name}_accuracy": float(accuracy),
            f"{model_name}_f1": float(f1),
            f"{model_name}_recall": float(recall),
            f"{model_name}_precision": float(precision),
            f"{model_name}_roc_auc": float(roc_auc)
        })

        # Save the models. to model registry/folder based on the validation set performance.
        log.info("Saving models to Model Registry...")
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['model_registry']['model_path'], f"{model_name}_acc_{accuracy:.3f}_roc_{roc_auc:.3f}.pkl")
       
        # Remove unnecessary model attributes.
        # log.info("Removing unnecessary model attributes..")
        for attr in ['X_train_', 'y_train_', "oob_score_", 'oob_decision_function']:
            if hasattr(model, attr):
                delattr(model, attr)

        # Save model using joblib without compression to reduce model size.
        joblib.dump(model, model_path)
        # Upload model to Neptune
        log_model(run, model_path, alias=model_name)

    # Get the best model to use for retraining:                                                                         
    # This is based on the balanced accuracy score.
    best_model_name = max(val_model_performance, key=lambda name: val_model_performance[name][1])
    best_val_model, _, _, _, _, _ = val_model_performance[best_model_name]


    # Define temp. model validtion
    records = []
    for name, (model, accuracy, f1, recall, precision, roc_auc) in val_model_performance.items():
        records.append({
            'model': name,
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'roc_auc': roc_auc
        })
    # Create dataframe from records.
    val_df_results = pd.DataFrame(records)

    # Save the validation results to csv.
    val_df_results.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), config['cv_result'], "val_model_performance.csv"), index=False)
    log.info(f"Best model based on validation set: {best_model_name} with accuracy: {val_model_performance[best_model_name][1]}")

    # Stop Neptune run
    stop_run(run)

    return best_val_model

    

def retrain_model(train_data: pd.DataFrame, val_data: pd.DataFrame, log, model, config):
    # To perform retraining by merging both train and val datasets.
    X_train_val = pd.concat([train_data, val_data])
    X_train_val = X_train_val.sort_values('timestamp').reset_index(drop=True)
    y_train_val = X_train_val['aqi']
    X_train_val = X_train_val.drop(['aqi', 'timestamp'])

    # Select best models from the evaluation on the validation set [e.g X_val].
    final_model = model.fit(X_train_val, y_train_val)
    path = f"{config['model_registry']['model_path']}/best_model.pkl"
    # Save with joblib.
    joblib.dump(final_model, path)

    return final_model




if __name__ == "__main__":
    pass
    # # Initialize neptune run.
    # run = init_neptune_run(config['neptune']['api_token'], config['neptune']['project_name'], config['neptune']['experiment_name'])
    
    # # Log the model and metrics.
    # log_model(run, "aqi_model", "aqi_model.pkl")
    # log_metrics(run, "metrics", ["accuracy", "f1", "recall", "precision", "roc_auc"])

    # Train the model.
    # train_model(train_df, val_df, log)

    # # Stop the run.
    # stop_run(run)
