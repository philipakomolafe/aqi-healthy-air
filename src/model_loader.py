# TODO: LOADING TRAINED MODELS FROM DISK.
import joblib
from loguru._logger import Logger

def load_model(model_path: str, log: Logger):
    """
    Load a trained model from the specified path.
    """

    try:
        log.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        log.success("Model loaded successfully...")
        return model

    except Exception as e:
        log.error(f"Error loading model from {model_path}: {e}")
        raise e