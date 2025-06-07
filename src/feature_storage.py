import os
import pandas as pd
from datetime import datetime

def save_features(data: pd.DataFrame, filename: str, output_dir: str, version: str | None = None):
    """
    Save processed features to disk with optional versioning.
    """
    if version is None:
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(filename)
    versioned_filename = f"{base}_v{version}{ext}"
    path = os.path.join(output_dir, versioned_filename)
    data.to_csv(path, index=False)
    return path
