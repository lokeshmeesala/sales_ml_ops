import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessingStrategy(DataStrategy):
    """
    Stragegy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchace_timestamp",
            ], axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_g"].fillna(data["product_length_g"].median(), inplace=True)
            data["product_height_g"].fillna(data["product_height_g"].median(), inplace=True)
            data["product_width_g"].fillna(data["product_width_g"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing the data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['revie_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=314)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing the data {e}")
            raise e

class DataCleaning:
    def __init__(self, data:pd.DataFrame, strategy:DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in divdata clean {e}")
            raise e