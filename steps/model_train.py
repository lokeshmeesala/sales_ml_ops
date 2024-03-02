import logging
import pandas as pd
from zenml import step

from src.model_dev import LinReg
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client

experiment_tracker_client = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker_client.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    abc
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinReg()
            # mlflow.sklearn.log_model(model)
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error in training model {e}")
        raise e