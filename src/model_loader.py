# TODO: LOADING TRAINED MODELS FROM DISK.
import joblib
import os
import zipfile
import tempfile
from pathlib import Path
from loguru._logger import Logger

    
def load_model(model_path: str, model_type: str, log: Logger):
    """
    Load the best model based on its type.
    
    Args:
        model_path: Path to model file (.pkl), directory (for DL models), or zip file (for DL models)
        model_type: Either 'classical' or 'deep_learning'
    
    Returns:
        Loaded model
    """
    try:
        if model_type == 'classical':
            import joblib
            log.info(f"Loading classical ML model from {model_path}")
            model = joblib.load(model_path)
            log.success("Classical model loaded successfully...")
            return model

        elif model_type == 'deep_learning':
            # Import here to avoid circular imports
            from .train import DeepLearningWrapper
            
            log.info(f"Loading deep learning model from {model_path}")
            
            # Check if it's a zip file or directory
            if model_path.endswith('.zip') and os.path.isfile(model_path):
                log.info("Detected zip file format for deep learning model")
                model = DeepLearningWrapper.load_from_zip(model_path)
            elif os.path.isdir(model_path):
                log.info("Detected directory format for deep learning model")
                model = DeepLearningWrapper.load_model(model_path)
            else:
                raise ValueError(f"Invalid deep learning model path: {model_path}. Expected directory or zip file.")
            
            log.success("Deep learning model loaded successfully...")
            return model
        
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'classical' or 'deep_learning'")
             
    except Exception as e:
        log.error(f"Error loading model from {model_path}: {e}")
        raise e


def auto_detect_and_load_model(model_path: str, log: Logger):
    """
    Automatically detect model type and load the appropriate model.
    
    Args:
        model_path: Path to model file, directory, or zip file
        log: Logger instance
    
    Returns:
        Tuple of (loaded_model, model_type)
    """
    try:
        log.info(f"Auto-detecting model type for {model_path}")
        
        if model_path.endswith('.pkl') and os.path.isfile(model_path):
            # Classical ML model
            log.info("Detected classical ML model (.pkl file)")
            model = load_model(model_path, 'classical', log)
            return model, 'classical'
        
        elif model_path.endswith('.zip') and os.path.isfile(model_path):
            # Deep learning model in zip format
            log.info("Detected deep learning model (.zip file)")
            model = load_model(model_path, 'deep_learning', log)
            return model, 'deep_learning'
        
        elif os.path.isdir(model_path):
            # Check if it's a deep learning model directory
            keras_model_path = os.path.join(model_path, 'keras_model.h5')
            metadata_path = os.path.join(model_path, 'metadata.pkl')
            
            if os.path.exists(keras_model_path) and os.path.exists(metadata_path):
                log.info("Detected deep learning model (directory format)")
                model = load_model(model_path, 'deep_learning', log)
                return model, 'deep_learning'
            else:
                raise ValueError(f"Directory {model_path} does not contain required deep learning model files")
        
        else:
            raise ValueError(f"Could not detect model type for {model_path}")
            
    except Exception as e:
        log.error(f"Error in auto-detection and loading: {e}")
        raise e


def load_from_neptune_download(downloaded_path: str, log: Logger):
    """
    Load model from Neptune download (could be zip or pkl).
    
    Args:
        downloaded_path: Path to downloaded model from Neptune
        log: Logger instance
    
    Returns:
        Tuple of (loaded_model, model_type)
    """
    try:
        log.info(f"Loading model from Neptune download: {downloaded_path}")
        return auto_detect_and_load_model(downloaded_path, log)
        
    except Exception as e:
        log.error(f"Error loading model from Neptune download: {e}")
        raise e


def extract_and_load_zip_model(zip_path: str, extract_to: str = None, log: Logger = None):
    """
    Extract zip file and load deep learning model.
    
    Args:
        zip_path: Path to zip file containing deep learning model
        extract_to: Directory to extract to (if None, uses temp directory)
        log: Logger instance
    
    Returns:
        Loaded model
    """
    try:
        if extract_to is None:
            extract_to = tempfile.mkdtemp()
        
        if log:
            log.info(f"Extracting zip model from {zip_path} to {extract_to}")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
        
        # Load from extracted directory
        from .train import DeepLearningWrapper
        model = DeepLearningWrapper.load_model(extract_to)
        
        if log:
            log.success("Zip model extracted and loaded successfully")
        
        return model
        
    except Exception as e:
        if log:
            log.error(f"Error extracting and loading zip model: {e}")
        raise e


def get_model_info_from_path(model_path: str) -> dict:
    """
    Extract model information from file/directory path.
    
    Args:
        model_path: Path to model
    
    Returns:
        Dictionary with model information
    """
    import re
    
    try:
        if model_path.endswith('.pkl'):
            # Classical ML model
            pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)\.pkl")
            filename = os.path.basename(model_path)
            match = pattern.search(filename)
            
            if match:
                return {
                    'model_name': match.group(1),
                    'accuracy': float(match.group(2)),
                    'roc_auc': float(match.group(3)),
                    'type': 'classical',
                    'format': 'pkl',
                    'path': model_path
                }
        
        elif model_path.endswith('.zip'):
            # Deep learning model zip
            pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)\.zip")
            filename = os.path.basename(model_path)
            match = pattern.search(filename)
            
            if match:
                return {
                    'model_name': match.group(1),
                    'accuracy': float(match.group(2)),
                    'roc_auc': float(match.group(3)),
                    'type': 'deep_learning',
                    'format': 'zip',
                    'path': model_path
                }
        
        elif os.path.isdir(model_path):
            # Deep learning model directory
            pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)")
            dirname = os.path.basename(model_path)
            match = pattern.search(dirname)
            
            if match:
                return {
                    'model_name': match.group(1),
                    'accuracy': float(match.group(2)),
                    'roc_auc': float(match.group(3)),
                    'type': 'deep_learning',
                    'format': 'directory',
                    'path': model_path
                }
        
        return {'error': 'Could not extract model information', 'path': model_path}
        
    except Exception as e:
        return {'error': str(e), 'path': model_path}


if __name__ == "__main__":
    # Example usage
    pass