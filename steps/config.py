from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Contains all model configration parameters
    """
    model_name: str = "LinearRegression"
     