# TODO: LOADING TRAINED MODELS FROM DISK.
import joblib
from loguru._logger import Logger

def load_emodel(model_path: str, log: Logger):
    """
    Load a trained model from the specified path.
    """

    try:
        log.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        return model

    except Exception as e:
        
        raise e

    
def load_model(model_path: str, model_type: str, log: Logger):
    """
    Load the best model based on its type.
    
    Args:
        model_path: Path to model file (.pkl) or directory (for DL models)
        model_type: Either 'classical' or 'deep_learning'
    
    Returns:
        Loaded model
    """
    try:

        if model_type == 'classical':
            import joblib
            log.success("Classical model loaded successfully...")
            return joblib.load(model_path)

        elif model_type == 'deep_learning':
            # Import here to avoid circular imports
            from .train import DeepLearningWrapper
            log.success("Deep learning model loaded successfully...")
            return DeepLearningWrapper.load_model(model_path)
    
             
    except Exception as e:
        log.error(f"Error loading model from {model_path}: {e}")