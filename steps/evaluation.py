import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.model_eval import RMSE, MSE, R2

import mlflow
from zenml.client import Client

experiment_tracker_client = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker_client.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, 'r2_score'],
                       Annotated[float, 'rmse_score']
                   ]:
    """
    abc
    """
    try:
        prediction = model.predict(X_test)
        mse = MSE()
        mse_score = mse.calculate_scores(y_test, prediction)
        mlflow.log_metric('mse_score', mse_score)

        r2 = R2()
        r2_score = r2.calculate_scores(y_test, prediction)
        mlflow.log_metric('r2_score', r2_score)

        rmse = RMSE()
        rmse_score = rmse.calculate_scores(y_test, prediction)
        mlflow.log_metric('rmse_score', rmse_score)

        return r2_score, rmse_score

    except Exception as e:
        logging.error(f"Error in evaluating the model {e}")
        raise e