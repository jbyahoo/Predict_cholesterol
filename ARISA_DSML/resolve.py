from ARISA_DSML.config import MODEL_NAME
import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from loguru import logger
from typing import Optional, List

def ensure_model_registered(client: MlflowClient) -> None:
    """Ensure the base model is registered in the registry."""
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException as e:
        if "not found" in str(e).lower():
            logger.info(f"Registering new model: {MODEL_NAME}")
            client.create_registered_model(MODEL_NAME)
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
        if "not found" in str(e).lower():
            logger.warning(f"Alias '{alias}' not found for model '{MODEL_NAME}'")
            return None
        logger.error(f"Registry access error: {str(e)}")
        raise

def get_latest_model_version(client: MlflowClient) -> Optional[ModelVersion]:
    """Get newest model version by creation timestamp."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None
    return sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]

def promote_model(client: MlflowClient, version: ModelVersion, alias: str) -> None:
    """Safely promote a model version to specified alias."""
    try:
        # Clear existing alias if exists
        existing = client.get_model_version_by_alias(MODEL_NAME, alias)
        client.delete_registered_model_alias(MODEL_NAME, alias)
    except MlflowException:
        pass  # Alias didn't exist
    
    client.set_registered_model_alias(MODEL_NAME, alias, version.version)
    logger.info(f"Promoted version {version.version} to {alias}")

def evaluate_challenger(
    client: MlflowClient, 
    champion: ModelVersion, 
    challenger: ModelVersion
) -> None:
    """Evaluate challenger against champion metrics."""
    champ_run = client.get_run(champion.run_id)
    chall_run = client.get_run(challenger.run_id)
    
    f1_champ = champ_run.data.metrics.get("f1_cv_mean", 0)
    f1_chall = chall_run.data.metrics.get("f1_cv_mean", 0)

    if f1_chall >= f1_champ:
        logger.info("Challenger outperformed champion - updating aliases")
        promote_model(client, challenger, "champion")
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
    else:
        logger.error("Challenger failed to outperform champion")
        raise ValueError("Challenger model metrics below champion threshold")

if __name__ == "__main__":
    client = MlflowClient(mlflow.get_tracking_uri())
    
    # Ensure model exists before operations
    ensure_model_registered(client)
    
    # Check existing aliases
    champion = get_model_by_alias(client, "champion")
    challenger = get_model_by_alias(client, "challenger")

    if not champion:
        if challenger:
            logger.info("Promoting existing challenger to champion")
            promote_model(client, challenger, "champion")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
        else:
            latest_version = get_latest_model_version(client)
            if latest_version:
                logger.info("No aliases found - promoting latest version to champion")
                promote_model(client, latest_version, "champion")
            else:
                logger.error("No model versions available in registry")
                raise RuntimeError("No deployable models found in registry")

    # Regular challenger evaluation flow
    if champion and challenger:
        evaluate_challenger(client, champion, challenger)
    elif champion:
        logger.info("No active challenger - continuing with current champion")