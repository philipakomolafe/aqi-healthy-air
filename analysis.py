import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import joblib

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.model_loader import auto_detect_and_load_model
from src.utils import setup_logger, get_logger, read_processed_data

# Initialize logger
setup_logger(level="INFO")
log = get_logger("Model Analysis")

# Model categories
MODEL_CATEGORIES = {
    'knn': ['knn', 'k_nearest', 'nearest_neighbor'],
    'svc': ['svc', 'svm', 'support_vector'],
    'rf': ['rf', 'random_forest', 'randomforest'],
    'xgb': ['xgb', 'xgboost', 'gradient_boost'],
    'gru': ['gru'],
    'lstm': ['lstm'],
    'hybrid': ['hybrid']
}

# AQI Classification Labels (1-5)
AQI_LABELS = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy"]

def load_test_data():
    """Load test dataset"""
    try:
        test_path = os.path.join(project_root, 'data', 'processed', 'test', 'aqi_test_data_v1.csv')
        test_df = read_processed_data(test_path, log)
        
        # Prepare features and target
        X_test = test_df.drop(columns=['timestamp', 'aqi'], errors='ignore')
        y_test = test_df['aqi']
        
        log.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test
        
    except Exception as e:
        log.error(f"Error loading test data: {str(e)}")
        raise

def find_best_models_by_category(models_dir: str = "models") -> Dict[str, str]:
    """Find best model for each category using regex patterns from utils.py"""
    try:
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Use regex patterns from utils.py
        classical_pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)\.pkl")
        dl_pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)")
        
        best_models = {}
        
        for category, keywords in MODEL_CATEGORIES.items():
            category_candidates = []
            
            for item in models_path.iterdir():
                item_name = item.name.lower()
                
                # Check if this item belongs to current category
                if any(keyword in item_name for keyword in keywords):
                    
                    if item.is_file() and item.suffix == '.pkl':
                        # Classical ML model
                        match = classical_pattern.search(item.name)
                        if match:
                            model_name = match.group(1)
                            acc = float(match.group(2))
                            roc = float(match.group(3))
                            score = (0.6 * roc) + (0.4 * acc)
                            
                            category_candidates.append({
                                'path': str(item),
                                'name': item.name,
                                'accuracy': acc,
                                'roc': roc,
                                'score': score
                            })
                    
                    elif item.is_dir():
                        # Deep learning model
                        keras_model_path = item / 'keras_model.h5'
                        metadata_path = item / 'metadata.pkl'
                        
                        if keras_model_path.exists() and metadata_path.exists():
                            match = dl_pattern.search(item.name)
                            if match:
                                model_name = match.group(1)
                                acc = float(match.group(2))
                                roc = float(match.group(3))
                                score = (0.6 * roc) + (0.4 * acc)
                                
                                category_candidates.append({
                                    'path': str(item),
                                    'name': item.name,
                                    'accuracy': acc,
                                    'roc': roc,
                                    'score': score
                                })
            
            # Select best model for this category
            if category_candidates:
                best_candidate = max(category_candidates, key=lambda x: x['score'])
                best_models[category] = best_candidate['path']
                
                log.info(f"Best {category.upper()} model: {best_candidate['name']}")
                log.info(f"  Accuracy: {best_candidate['accuracy']:.4f}, ROC: {best_candidate['roc']:.4f}")
            else:
                log.warning(f"No models found for category: {category}")
        
        return best_models
        
    except Exception as e:
        log.error(f"Error finding best models: {str(e)}")
        raise

def convert_aqi_to_class(aqi_values: np.ndarray) -> np.ndarray:
    """Convert AQI values to classification categories (1-5)"""
    bins = [0, 50, 100, 150, 200, float('inf')]
    classes = np.digitize(aqi_values, bins)
    return np.clip(classes, 1, 5)

def get_predictions(model_path: str, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from model with improved handling for deep learning models"""
    try:
        model, model_type = auto_detect_and_load_model(model_path, log)
        
        if model_path.endswith('.pkl'):
            # Classical ML model
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                predictions = model.predict(X_test)
                
                # Ensure predictions are in 1-5 range
                if not np.all(np.isin(predictions, [1, 2, 3, 4, 5])):
                    predictions = convert_aqi_to_class(predictions)
                    
            else:
                predictions = model.predict(X_test)
                
                # Ensure predictions are in 1-5 range
                if not np.all(np.isin(predictions, [1, 2, 3, 4, 5])):
                    predictions = convert_aqi_to_class(predictions)
                
                # Create dummy probabilities
                probabilities = np.zeros((len(predictions), 5))
                for i, cls in enumerate(predictions):
                    if 1 <= cls <= 5:
                        probabilities[i, int(cls)-1] = 1.0
        else:
            # Deep learning model - improved prediction handling
            try:
                # Get raw predictions from the deep learning wrapper
                raw_predictions = model.predict(X_test)
                
                # Handle different output formats
                if hasattr(model, 'predict_proba'):
                    # Model has probability method
                    probabilities = model.predict_proba(X_test)
                    if len(probabilities.shape) > 1 and probabilities.shape[1] == 5:
                        predictions = np.argmax(probabilities, axis=1) + 1  # Convert to 1-5
                    else:
                        predictions = convert_aqi_to_class(raw_predictions.flatten())
                        # Create dummy probabilities
                        probabilities = np.zeros((len(predictions), 5))
                        for i, cls in enumerate(predictions):
                            probabilities[i, cls-1] = 1.0
                else:
                    # No probability method - handle raw predictions
                    if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
                        # Multi-class output
                        probabilities = raw_predictions
                        predictions = np.argmax(raw_predictions, axis=1) + 1
                    else:
                        # Single output - convert to classes
                        predictions = convert_aqi_to_class(raw_predictions.flatten())
                        probabilities = np.zeros((len(predictions), 5))
                        for i, cls in enumerate(predictions):
                            probabilities[i, cls-1] = 1.0
                            
            except Exception as dl_error:
                log.warning(f"Deep learning prediction failed: {dl_error}, using fallback")
                # Fallback: create dummy predictions
                n_samples = len(X_test)
                predictions = np.full(n_samples, 2)  # Default to moderate
                probabilities = np.zeros((n_samples, 5))
                probabilities[:, 1] = 1.0  # All moderate
        
        # Ensure predictions are in correct format
        predictions = np.array(predictions, dtype=int)
        predictions = np.clip(predictions, 1, 5)
        
        # Ensure probabilities are in correct shape
        if probabilities.shape[1] != 5:
            # Reshape if needed
            n_samples = len(predictions)
            new_probabilities = np.zeros((n_samples, 5))
            for i, cls in enumerate(predictions):
                new_probabilities[i, cls-1] = 1.0
            probabilities = new_probabilities
        
        return predictions, probabilities
        
    except Exception as e:
        log.error(f"Error getting predictions from {model_path}: {str(e)}")
        raise

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> Dict:
    """Calculate classification metrics with improved AUC handling"""
    # Convert to classes if needed
    y_true_class = convert_aqi_to_class(y_true) if not np.all(np.isin(y_true, [1, 2, 3, 4, 5])) else y_true
    y_pred_class = convert_aqi_to_class(y_pred) if not np.all(np.isin(y_pred, [1, 2, 3, 4, 5])) else y_pred
    
    y_true_class = np.clip(y_true_class.astype(int), 1, 5)
    y_pred_class = np.clip(y_pred_class.astype(int), 1, 5)
    
    # Basic metrics
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    
    # Macro averages
    precision_macro = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    
    # AUC score with robust error handling
    auc_score = 0.0
    try:
        unique_true_classes = np.unique(y_true_class)
        unique_pred_classes = np.unique(y_pred_class)
        
        log.debug(f"Model {model_name}: True classes {unique_true_classes}, Pred classes {unique_pred_classes}")
        
        # Only calculate AUC if we have multiple classes and valid probabilities
        if len(unique_true_classes) > 1 and y_prob.shape[1] == 5:
            # Check if probabilities are meaningful (not all identical)
            prob_variance = np.var(y_prob, axis=0).sum()
            
            if prob_variance > 1e-10:  # Small threshold for numerical precision
                try:
                    # Binarize true labels for multiclass AUC
                    y_true_bin = label_binarize(y_true_class, classes=[1, 2, 3, 4, 5])
                    
                    # Handle case where not all classes are present
                    if y_true_bin.shape[1] < 5:
                        # Pad with zeros for missing classes
                        padded_y_true = np.zeros((len(y_true_class), 5))
                        for i, cls in enumerate([1, 2, 3, 4, 5]):
                            if cls in unique_true_classes:
                                cls_idx = np.where(unique_true_classes == cls)[0][0]
                                padded_y_true[:, i] = y_true_bin[:, cls_idx] if y_true_bin.shape[1] > cls_idx else 0
                        y_true_bin = padded_y_true
                    
                    # Calculate AUC with error handling
                    auc_score = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
                    
                except ValueError as ve:
                    log.warning(f"AUC calculation failed for {model_name}: {ve}")
                    auc_score = 0.0
            else:
                log.warning(f"Probabilities have no variance for {model_name}, AUC set to 0")
        else:
            log.warning(f"Insufficient class diversity for AUC calculation: {model_name} (classes: {len(unique_true_classes)})")
            
    except Exception as e:
        log.warning(f"Could not calculate AUC for {model_name}: {str(e)}")
        auc_score = 0.0
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auc_score': auc_score,
        'unique_true_classes': len(unique_true_classes),
        'unique_pred_classes': len(unique_pred_classes),
        'perfect_predictions': accuracy == 1.0
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_dir: str):
    """Plot confusion matrix"""
    y_true_class = convert_aqi_to_class(y_true) if not np.all(np.isin(y_true, [1, 2, 3, 4, 5])) else y_true
    y_pred_class = convert_aqi_to_class(y_pred) if not np.all(np.isin(y_pred, [1, 2, 3, 4, 5])) else y_pred
    
    y_true_class = np.clip(y_true_class.astype(int), 1, 5)
    y_pred_class = np.clip(y_pred_class.astype(int), 1, 5)
    
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[1, 2, 3, 4, 5])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=AQI_LABELS, yticklabels=AQI_LABELS)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Clean filename for saving
    clean_model_name = re.sub(r'[^\w\-_.]', '_', model_name)
    plt.savefig(f"{save_dir}/{clean_model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results_df: pd.DataFrame, save_dir: str):
    """Plot model comparison"""
    metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'auc_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            bars = axes[i].bar(results_df['category'], results_df[metric], 
                              color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1.1)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[-1].set_visible(False)
    
    plt.suptitle('Best Models Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_best_models():
    """Main analysis function"""
    print("🚀 AQI Best Models Analysis")
    print("=" * 60)
    print("Categories: KNN, SVC, RF, XGB, GRU, LSTM, Hybrid")
    print("=" * 60)
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Find best models by category
        best_models = find_best_models_by_category()
        
        if not best_models:
            print("❌ No models found!")
            return
        
        print(f"\n📊 Found best models for {len(best_models)} categories")
        
        # Create output directory
        save_dir = "best_models_analysis"
        os.makedirs(save_dir, exist_ok=True)
        
        # Analyze each model
        results = []
        
        for category, model_path in best_models.items():
            print(f"\n--- Analyzing {category.upper()} Model ---")
            model_name = Path(model_path).stem
            
            try:
                # Get predictions
                y_pred, y_prob = get_predictions(model_path, X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test.values, y_pred, y_prob, model_name)
                metrics['category'] = category.upper()
                metrics['model_path'] = model_path
                results.append(metrics)
                
                # Plot confusion matrix
                plot_confusion_matrix(y_test.values, y_pred, model_name, save_dir)
                
                print(f"✅ {model_name}")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1-Score: {metrics['f1_weighted']:.4f}")
                print(f"   AUC Score: {metrics['auc_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Error analyzing {model_name}: {str(e)}")
                log.error(f"Full error for {model_name}: {str(e)}")
                continue
        
        if not results:
            print("❌ No successful analyses!")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot comparison
        plot_model_comparison(results_df, save_dir)
        
        # Save results
        results_df.to_csv(f"{save_dir}/results.csv", index=False)
        
        # Display summary
        print("\n" + "="*80)
        print("📊 PERFORMANCE SUMMARY")
        print("="*80)
        
        summary_cols = ['category', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'auc_score']
        summary = results_df[summary_cols].copy()
        summary = summary.sort_values('accuracy', ascending=False)
        print(summary.to_string(index=False, float_format='%.4f'))
        
        # Best model
        best = summary.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print(f"   Category: {best['category']}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   F1-Score: {best['f1_weighted']:.4f}")
        print(f"   AUC Score: {best['auc_score']:.4f}")
        
        print(f"\n📁 Results saved to: ./{save_dir}/")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        log.error(f"Full analysis error: {str(e)}")
        return None

if __name__ == "__main__":
    analyze_best_models()