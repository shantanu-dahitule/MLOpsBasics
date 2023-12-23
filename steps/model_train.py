import logging
import pandas as pd
from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train a model on the given data.
    """
    pass