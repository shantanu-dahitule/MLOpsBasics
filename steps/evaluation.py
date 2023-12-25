import pandas as pd
import logging
from zenml import step
from src.evaluation import MSE, R2
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml.cli import Client
import mlflow

experiment_tracker = Client().activate_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
) -> Tuple[Annotated[float, "r2_score"],
           Annotated[float, "mse"]
]:
    """
    Evaluate the model on the given data.
    """
    try:
        predictions = model.predict(X_test)
        mse = MSE().calculate_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        r2_score = R2().calculate_score(y_test, predictions)
        mlflow.log_metric("r2_score", r2_score)

        return mse, r2_score
    except Exception as e:
        logging.error("Error in evaluating the model {}".format(e))
        raise e