import pandas as pd
import logging
from zenml import step
from src.evaluation import MSE, R2
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
@step
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
        r2_score = R2().calculate_score(y_test, predictions)

        return mse, r2_score
    except Exception as e:
        logging.error("Error in evaluating the model {}".format(e))
        raise e