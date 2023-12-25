import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining the evaluation interface
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray ,y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy for Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation Strategy for R2
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 {}".format(e))
            raise e