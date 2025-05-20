from ARISA_DSML.config import MODEL_NAME, MODEL_PATH
import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from loguru import logger
from typing import Optional, List
import os

def ensure_model_registered(client: MlflowClient) -> None:
    """Ensure the base model is registered and has initial version."""
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException as e:
        if "not found" in str(e).lower():
            logger.info(f"Registering new model: {MODEL_NAME}")
            # Create registered model
            client.create_registered_model(MODEL_NAME)
            
            # Log initial model version
            with mlflow.start_run() as run:
                mlflow.log_artifact(MODEL_PATH, "model")
                result = mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model",
                    name=MODEL_NAME
                )
                
                # Set initial aliases
                client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
                client.set_registered_model_alias(MODEL_NAME, "challenger", result.version)
                
            logger.success(f"Initial version {result.version} registered with aliases")
        else:
            logger.error(f"Model registry error: {str(e)}")
            raise

def get_model_by_alias(
    client: MlflowClient,
    alias: str = "champion"
) -> Optional[ModelVersion]:
    """Retrieve a model version by its alias."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.warning(f"Alias '{alias}' not found for model '{MODEL_NAME}'")
            return None
        logger.error(f"Registry access error: {str(e)}")
        raise

def get_latest_model_version(client: MlflowClient) -> Optional[ModelVersion]:
    """Get newest model version by creation timestamp."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None
    return max(versions, key=lambda v: v.creation_timestamp)

def promote_model(client: MlflowClient, version: ModelVersion, alias: str) -> None:
    """Safely promote a model version to specified alias."""
    try:
        client.set_registered_model_alias(MODEL_NAME, alias, version.version)
        logger.info(f"Promoted version {version.version} to {alias}")
    except MlflowException as e:
        logger.error(f"Promotion failed: {str(e)}")
        raise

def evaluate_challenger(
    client: MlflowClient, 
    champion: ModelVersion, 
    challenger: ModelVersion
) -> None:
    """Evaluate challenger against champion metrics."""
    try:
        champ_run = client.get_run(champion.run_id)
        chall_run = client.get_run(challenger.run_id)
    except MlflowException as e:
        logger.error(f"Failed to get run details: {str(e)}")
        raise

    f1_champ = champ_run.data.metrics.get("f1_cv_mean", 0)
    f1_chall = chall_run.data.metrics.get("f1_cv_mean", 0)

    if f1_chall >= f1_champ:
        logger.info("Challenger outperformed champion - updating aliases")
        promote_model(client, challenger, "champion")
        promote_model(client, challenger, "challenger")
    else:
        logger.error("Challenger failed to outperform champion")
        raise ValueError("Challenger model metrics below champion threshold")

if __name__ == "__main__":
    # Verify model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    client = MlflowClient(mlflow.get_tracking_uri())
    
    # Ensure model exists with initial version
    ensure_model_registered(client)
    
    # Check existing aliases
    champion = get_model_by_alias(client, "champion")
    challenger = get_model_by_alias(client, "challenger")

    # Handle first-run scenario
    if not champion and not challenger:
        latest_version = get_latest_model_version(client)
        if latest_version:
            logger.info("Initial setup - promoting latest version to both aliases")
            promote_model(client, latest_version, "champion")
            promote_model(client, latest_version, "challenger")
            champion = challenger = latest_version
        else:
            logger.error("No model versions available after registration")
            raise RuntimeError("Model registration failed")

    # Normal evaluation flow
    if champion and challenger:
        if champion.version != challenger.version:
            evaluate_challenger(client, champion, challenger)
        else:
            logger.info("Champion and challenger are same version - no evaluation needed")
    else:
        logger.info("Single model version active - no challenger to evaluate")