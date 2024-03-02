import click
from rich import print
from typing import cast
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_delployment_pipeline

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-C",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="""Optionally you can choose to only run the deployment
    pipeline to train and deploy a model ('deploy'), or to
    only run a prediction against the deployed model
    ('predict). By default both will be run ('deploy_and_predict')."""
)

@click.option(
    "--min-accuracy",
    default=0.001,
    help="Minimum accuracy required to deploy the model"
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    print("mlflow_model_deployer_component",mlflow_model_deployer_component)
    if deploy:
        continuous_delployment_pipeline(
            data_path=".\data\orders_products_reviews_dataset.csv",
            min_accuracy=min_accuracy,
            workers=1,
            timeout=60)
    
    # if predict:
    #     # inference_pipeline()
        
    print("You can run:\n"
          f"[italic green] mlflow ui --backend-store-uri {get_tracking_uri()}"
          "[italic green]\n ...to inspect your experiment runs within the MLflow UI. \n"
          "You can find your runs tracked within the `mlflow_example_pipeline` experiment.")
    

    # fetch existing services with same pipeline name, step name and model name
    existing_service = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )
    print("existing_service",existing_service)
    if existing_service:
        service = cast(MLFlowDeploymentService, existing_service[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, re-run the same command and supply the "
                f"`--stop-service` argument."
            )
        else:
            print("NO MLflow prediction server found")

if __name__ == "__main__":
    run_deployment()