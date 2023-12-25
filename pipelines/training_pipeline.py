from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True) # Try checking chache as false
def train_pipeline(data_path: str) -> pd.DataFrame:
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test =  clean_data(df)
    model = train_model(X_train, y_train)
    mse, r2_Score = evaluate_model(model, X_test, y_test)
