import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression

class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, X_train, y_train):
        """
        abc
        """
        pass

class LinReg(Model):
    """
    abc
    """
    def train(self, X_train, y_train, **kwargs):
        """
        abc
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        
        except Exception as e:
            logging.error(f"Error in training LinReg {e}")
            raise e