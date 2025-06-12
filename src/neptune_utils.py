# NOTE: This file is to initialize there neptune utilities for the project.
import os
import neptune
from dotenv import load_dotenv


load_dotenv()
def init_neptune_run(project_name, api_token, tags, params) -> neptune.Run:
    """
    Initialize a Neptune run.
    """
    run = neptune.init_run(
        project=project_name or os.getenv("NEPTUNE_PROJECT_NAME"),
        api_token=api_token or os.getenv("NEPTUNE_API_TOKEN"),
        tags = tags or [],
        )

    if params:
        run['parameters'] = params
    return run


def log_metrics(run: neptune.Run, metrics_dict: dict[str, float], prefix='metrics'):
    """
    Log metrics to Neptune under given namespace.
    """
    for metric, value in metrics_dict.items():
            run[f"{prefix}/{metric}"].append(value)

def log_model(run: neptune.Run, model_path, alias = "best"):
    """
    Upload model file (.joblib, .pkl or .h5) to Neptune 
    """

    run[f'model/{alias}'].upload(model_path)


def stop_run(run):
    """
    Stops the current Neptune run.
    """
    run.stop()