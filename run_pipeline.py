from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="./data/olist_customers_dataset.csv")

"""
To run the backend of MLFLOW

mlflow ui --backend-store-uri "file:C:\Users\Shantanu Dahitule\AppData\Roaming\zenml\local_stores\183fa5e6-9489-4973-bd97-dae16bb1b6f5\mlruns"
"""