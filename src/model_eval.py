import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    abc
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred: np.ndarray):
        """
        abc
        """
        pass

class MSE(Evaluation):
    """
    abc
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        abc
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE {e}")
            raise e

class R2(Evaluation):
    """
    abc
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        abc
        """
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 {e}")
            raise e
                
class RMSE(Evaluation):
    """
    abc
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        abc
        """
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=True)
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE {e}")
            raise e