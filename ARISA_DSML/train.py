from pathlib import Path
from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
from mlflow.client import MlflowClient
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from azure.identity import DefaultAzureCredential
import os

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash
import nannyml as nml

SQL_CONNECTION_STRING = (
    f"mssql+pyodbc://{os.environ['AZURE_DBUSERNAME']}:{os.environ['AZURE_DBUSERPASS']}@{os.environ['AZURE_DBSERVERNAME']}.database.windows.net:1433/{os.environ['AZURE_DBNAME']}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
mlflow.set_tracking_uri(SQL_CONNECTION_STRING)

credential = DefaultAzureCredential()


def run_hyperopt(X_train: pd.DataFrame, y_train: pd.DataFrame, test_size: float = 0.25, n_trials: int = 20, overwrite: bool = False) -> str | Path:
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.trial.Trial) -> float:
            with mlflow.start_run(nested=True):
                params = {
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
                }
                model = CatBoostClassifier(**params, verbose=0)
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    early_stopping_rounds=50,
                )
                mlflow.log_params(params)
                preds = model.predict(X_val_opt)
                probs = model.predict_proba(X_val_opt)
                f1 = f1_score(y_val_opt, preds)
                logloss = log_loss(y_val_opt, probs)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("logloss", logloss)
                return logloss

        study.optimize(objective, n_trials=n_trials)
        joblib.dump(study.best_params, best_params_path)
    return best_params_path

def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict | None,
    artifact_name: str = MODEL_NAME,
    cv_results=None,
) -> tuple[str | Path]:
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}
    params = {k: v for k, v in params.items() if k not in ["feature_columns"]}

    with mlflow.start_run():
        model = CatBoostClassifier(**params, verbose=True)
        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            use_best_model=False,
            plot=True,
        )
        mlflow.log_params(params)
        mlflow.log_param("feature_columns", list(X_train.columns))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)

        if isinstance(cv_results, pd.DataFrame):
            required_cols_f1 = {"test-F1-mean"}
            required_cols_logloss = {"iterations", "test-Logloss-mean", "test-Logloss-std"}
            if required_cols_f1.issubset(cv_results.columns):
                cv_metric_mean = cv_results["test-F1-mean"].mean()
                mlflow.log_metric("f1_cv_mean", cv_metric_mean)
            else:
                logger.warning("cv_results missing required F1 columns for metric logging.")
            if {"iterations", "test-F1-mean", "test-F1-std"}.issubset(cv_results.columns):
                fig1 = plot_error_scatter(
                    df_plot=cv_results,
                    name="Mean F1 Score",
                    title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                    xtitle="Training Steps",
                    ytitle="Performance Score",
                    yaxis_range=[0.5, 1.0],
                )
                mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")
            else:
                logger.warning("cv_results missing required columns for F1 plot.")
            if required_cols_logloss.issubset(cv_results.columns):
                fig2 = plot_error_scatter(
                    cv_results,
                    x="iterations",
                    y="test-Logloss-mean",
                    err="test-Logloss-std",
                    name="Mean logloss",
                    title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                    xtitle="Training Steps",
                    ytitle="Logloss",
                )
                mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")
            else:
                logger.warning("cv_results missing required columns for Logloss plot.")
        else:
            logger.warning("cv_results is not a DataFrame. Skipping cv metrics and plots.")

        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            input_example=X_train,
            registered_model_name=MODEL_NAME,
        )
        client = MlflowClient(mlflow.get_tracking_uri())
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if latest_versions:
            model_info = latest_versions[0]
            client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
            client.set_model_version_tag(
                name=model_info.name,
                version=model_info.version,
                key="git_sha",
                value=get_git_commit_hash(),
            )
        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(params, model_params_path)

        reference_df = X_train.copy()
        reference_df["prediction"] = model.predict(X_train)
        reference_df["predicted_probability"] = [p[1] for p in model.predict_proba(X_train)]
        reference_df[target] = y_train
        chunk_size = 20

        udc = nml.UnivariateDriftCalculator(
            column_names=[col for col in X_train.columns if col != "PassengerId"],
            chunk_size=chunk_size,
        )
        udc.fit(reference_df.drop(columns=["prediction", target, "predicted_probability"], errors="ignore"))

        estimator = nml.CBPE(
            problem_type="classification_binary",
            y_pred_proba="predicted_probability",
            y_pred="prediction",
            y_true=target,
            metrics=["roc_auc"],
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")

        mlflow.log_artifact(MODELS_DIR / "udc.pkl")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl")

    return (model_path, model_params_path)

def train_cv(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict, eval_metric: str = "F1", n: int = 5) -> str | Path:
    params["eval_metric"] = eval_metric
    params["loss_function"] = "Logloss"
    data = Pool(X_train, y_train)
    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
        plot=True,
    )
    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)
    return cv_output_path

def plot_error_scatter(
        df_plot: pd.DataFrame,
        x: str = "iterations",
        y: str = "test-F1-mean",
        err: str = "test-F1-std",
        name: str = "",
        title: str = "",
        xtitle: str = "",
        ytitle: str = "",
        yaxis_range: list[float] | None = None,
) -> None:
    fig = go.Figure()
    if not len(name):
        name = y
    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[x], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y] + df_plot[err], df_plot[y] - df_plot[err][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )
    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )
    fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig

def get_or_create_experiment(experiment_name: str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)

if __name__ == "__main__":
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train_choloesterol.xlsx")
    y_train = df_train.pop(target)
    X_train = df_train

    experiment_name = "cholesterol_hyperparam_tuning"
    get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    best_params_path = run_hyperopt(X_train, y_train)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, params)
    cv_results = pd.read_csv(cv_output_path)

    experiment_name = "cholesterol_full_training"
    get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    model_path, model_params_path = train(X_train, y_train, params, cv_results=cv_results)
