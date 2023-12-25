import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a model on the given data.
    """
    model = None
    if config.model_name == "LinearRegression":
        model = LinearRegressionModel().train(X_train, y_train)
        return model
    else:
        raise ValueError("Model not {} not supported".format(config.model_name))
    
    
    