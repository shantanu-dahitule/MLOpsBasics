import logging
import pandas as pd

from zenml import step

class IngestData:
    def __init__(self, path: str):
        self.path = path
    
    def get_data(self):
        logging.info(f'Ingesting data from {self.path}')
        return pd.read_csv(self.path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a given path.

    Args:
        data_path: Path to the data.
    Returns:
        Dataframe with the data.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error ingseting data: {e}')
        raise e
    
    