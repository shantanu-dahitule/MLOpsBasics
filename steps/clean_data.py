import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessingStrategy
@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:#pd.DataFrame:
    
    """
    Cleans the data and divide into train and test

    Args:
    df : Raw data

    Returns:
        X_train : Training data
        X_test : Testing data
        y_train : Training labels
        y_test : Testing labels
    """
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(data = df, strategy=process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(data = processed_data, strategy=divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning is Complete...")
    except Exception as e:
        logging.error("Error in cleaning the data {}".format(e))
        raise e
    
