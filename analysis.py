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

def extract_metrics_from_name(model_name: str) -> Tuple[float, float]:
    """Extract accuracy and AUC score from model filename"""
    try:
        # Look for patterns like acc_0.85, accuracy_0.85, auc_0.90, roc_0.90
        acc_pattern = r'(?:acc|accuracy)[_-]?(\d+\.?\d*)'
        auc_pattern = r'(?:auc|roc)[_-]?(\d+\.?\d*)'
        
        acc_match = re.search(acc_pattern, model_name.lower())
        auc_match = re.search(auc_pattern, model_name.lower())
        
        accuracy = float(acc_match.group(1)) if acc_match else 0.0
        auc_score = float(auc_match.group(1)) if auc_match else 0.0
        
        # Convert if values are in percentage format (>1)
        if accuracy > 1:
            accuracy = accuracy / 100
        if auc_score > 1:
            auc_score = auc_score / 100
            
        return accuracy, auc_score
        
    except Exception as e:
        log.warning(f"Could not extract metrics from {model_name}: {e}")
        return 0.0, 0.0

def find_best_models_by_category(models_dir: str = "models") -> Dict[str, str]:
    """Find best model for each category based on filename metrics"""
    try:
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find all model files and directories
        all_model_paths = []
        
        # Classical ML models (.pkl files)
        pkl_files = list(models_path.glob("*.pkl"))
        all_model_paths.extend([str(p) for p in pkl_files])
        
        # Deep learning models (directories)
        for item in models_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if any(item.glob("*.h5")) or any(item.glob("keras_model.h5")):
                    all_model_paths.append(str(item))
        
        log.info(f"Found {len(all_model_paths)} total models")
        
        # Categorize and find best models
        best_models = {}
        
        for category, keywords in MODEL_CATEGORIES.items():
            category_models = []
            
            for model_path in all_model_paths:
                model_name = Path(model_path).stem.lower()
                
                # Check if model belongs to this category
                if any(keyword in model_name for keyword in keywords):
                    accuracy, auc_score = extract_metrics_from_name(model_name)
                    category_models.append({
                        'path': model_path,
                        'name': Path(model_path).stem,
                        'accuracy': accuracy,
                        'auc_score': auc_score,
                        'combined_score': accuracy + auc_score  # Simple combined metric
                    })
            
            if category_models:
                # Sort by combined score (accuracy + auc)
                category_models.sort(key=lambda x: x['combined_score'], reverse=True)
                best_model = category_models[0]
                best_models[category] = best_model['path']
                
                log.info(f"Best {category.upper()} model: {best_model['name']}")
                log.info(f"  Accuracy: {best_model['accuracy']:.4f}, AUC: {best_model['auc_score']:.4f}")
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
    """Get predictions from model"""
    try:
        model, model_type = auto_detect_and_load_model(model_path)
        
        if model_path.endswith('.pkl'):
            # Classical ML model
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                predictions = model.predict(X_test)
            else:
                predictions = model.predict(X_test)
                # Create dummy probabilities
                probabilities = np.zeros((len(predictions), 5))
                for i, cls in enumerate(predictions):
                    if 1 <= cls <= 5:
                        probabilities[i, int(cls)-1] = 1.0
        else:
            # Deep learning model
            predictions = model.predict(X_test.values)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class output
                probabilities = predictions
                predictions = np.argmax(predictions, axis=1) + 1
            else:
                # Single output - convert to classes
                predictions = convert_aqi_to_class(predictions.flatten())
                probabilities = np.zeros((len(predictions), 5))
                for i, cls in enumerate(predictions):
                    probabilities[i, cls-1] = 1.0
        
        return predictions, probabilities
        
    except Exception as e:
        log.error(f"Error getting predictions from {model_path}: {str(e)}")
        raise

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> Dict:
    """Calculate classification metrics"""
    # Convert to classes if needed
    y_true_class = convert_aqi_to_class(y_true) if not np.all(np.isin(y_true, [1, 2, 3, 4, 5])) else y_true
    y_pred_class = convert_aqi_to_class(y_pred) if not np.all(np.isin(y_pred, [1, 2, 3, 4, 5])) else y_pred
    
    y_true_class = np.clip(y_true_class, 1, 5)
    y_pred_class = np.clip(y_pred_class, 1, 5)
    
    # Basic metrics
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    
    # Macro averages
    precision_macro = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    
    # AUC score
    try:
        if y_prob.shape[1] == 5:
            y_true_bin = label_binarize(y_true_class, classes=[1, 2, 3, 4, 5])
            auc_score = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
        else:
            auc_score = 0.0
    except:
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
        'auc_score': auc_score
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_dir: str):
    """Plot confusion matrix"""
    y_true_class = convert_aqi_to_class(y_true) if not np.all(np.isin(y_true, [1, 2, 3, 4, 5])) else y_true
    y_pred_class = convert_aqi_to_class(y_pred) if not np.all(np.isin(y_pred, [1, 2, 3, 4, 5])) else y_pred
    
    y_true_class = np.clip(y_true_class, 1, 5)
    y_pred_class = np.clip(y_pred_class, 1, 5)
    
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[1, 2, 3, 4, 5])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=AQI_LABELS, yticklabels=AQI_LABELS)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
        
        summary = results_df[['category', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'auc_score']].copy()
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
        return None

if __name__ == "__main__":
    analyze_best_models()