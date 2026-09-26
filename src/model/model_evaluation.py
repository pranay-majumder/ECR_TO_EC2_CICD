# Model Evaluation

import os
import json
import pickle
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import yaml

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


## Use it when you want to run the code in a local environment. It will set the MLflow tracking URI to DagsHub and initialize DagsHub with the specified repository details.
# mlflow.set_tracking_uri("https://dagshub.com/pranay-majumder/Docker_Pipeline_CI_CD.mlflow")
# dagshub.init(repo_owner="pranay-majumder", repo_name="Docker_Pipeline_CI_CD", mlflow=True)


# MLflow + DagsHub
# Set up DagsHub Credentials for MLflow Tracking (Usefull for GitHub Actions CI/CD Pipeline)
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "pranay-majumder"
repo_name = "Docker_Pipeline_CI_CD"

# # Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# Logging
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(file_path):
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters loaded from %s", file_path)
        return params

    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        raise

def load_model(file_path):
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }

        logger.debug("Model evaluation completed")
        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise


def save_model_info(run_id, model_id, model_path, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump({"run_id": run_id, "model_id": model_id, "model_path": model_path}, file, indent=4)

        logger.debug("Model info saved to %s", file_path)

    except Exception as e:
        logger.error("Error saving model info: %s", e)
        raise


def main():
    try:
        experiment_name="CI_Pipeline"
        mlflow.set_experiment(experiment_name)

        experiment = mlflow.get_experiment_by_name(experiment_name)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id]
        )

        next_number = len(runs) + 1

        with mlflow.start_run(run_name=f"Bow_LOR_{next_number}") as run:
            print("Run name:", mlflow.active_run().data.tags["mlflow.runName"])

            # Load model and Test Data
            model = load_model("./models/model.pkl")
            test_data = load_data("./data/feature_data/test_bow.csv")

            X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

            # Evaluate Model
            metrics = evaluate_model(model, X_test, y_test)

            # Save Metrics in Local File
            save_metrics(metrics, "reports/metrics.json")

            # Log Metrics and Parameters in DagsHub (MLflow)
            mlflow.log_metrics(metrics)

            # Log all model Parameters in DagsHub (MLflow)
            mlflow.log_params(model.get_params())

            # Log params.yaml parameters
            params = load_params("params.yaml")

            ## It should be in Dict format. So, we need to convert it into Dict format.
            mlflow.log_params({
                   "test_size": params["data_ingestion"]["test_size"],
                   "max_features": params["feature_engineering"]["max_features"]
                })

            # Log model
            # Later we need "model_id" for Load model from "Model Registory" of Mlflow.
            # For "Inference or Prediction" we need to load the Model.
            model_info=mlflow.sklearn.log_model(model, "model")
            model_id = model_info.model_id

            # Save and log model information
            save_model_info(run.info.run_id, model_id, "model", "reports/model_info.json")

            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/model_info.json")
            mlflow.log_artifact("model_evaluation_errors.log")

            print("Model evaluation completed:", metrics)

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()