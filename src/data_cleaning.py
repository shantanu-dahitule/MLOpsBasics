import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split
from typing import Union
class DataStrategy(ABC):
    """
    Abstract Class Defining to handle the data 
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame, pd.Series]:
        pass
class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy to handle the data preprocessing
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        """
        logging.info("Preprocessing the data")
        try:
            data = data.dropna([
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(e)
            raise e
           
class DataDivideStrategy(DataStrategy):
    """
    Strategy to divide the data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing the data {}".format(e))
            raise e
"""
This is imp class which calls the Abstract Classes
"""
class DataCleaning:
    """
    Class for cleaning the data which process the data and divide into train and test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy)->None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling the data {}".format(e))
            raise e

"""
 How to use the class
"""       
# if __name__ == "__main__":
#     #data/olist_customers_dataset.csv
#     data = pd.read_csv("../data/olist_order_items_dataset.csv")
#     data_cleaning = DataCleaning(data=data, strategy=DataPreProcessingStrategy())
#     data_cleaning.handle_data()
