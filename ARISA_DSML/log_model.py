import mlflow
with mlflow.start_run():
    mlflow.log_artifact('models/cholesterol-pred-bclass.cbm', artifact_path='models')