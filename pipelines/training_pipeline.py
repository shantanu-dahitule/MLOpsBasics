from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline # Try checking chache as false
def train_pipeline(data_path: str) -> pd.DataFrame:
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
