from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

class Model(ABC):
    """
    Abstract Class for all model
    """
    def train(self, X_train, y_train):
        pass


class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        """
        try: 
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model Trained")
            return model
        except Exception as e:
            logging.error("Error in training the model {}".format(e))
            raise e
        

    