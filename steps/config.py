from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    abc
    """
    model_name: str = "LinearRegression"