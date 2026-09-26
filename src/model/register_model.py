# Register Model

import json
import logging
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import os

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

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# Logging
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path):
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model information loaded from %s", file_path)
        return model_info
    except Exception as e:
        logger.error("Error loading model information: %s", e)
        raise


def register_model(model_id, model_name):
    try:
        model_uri = f"models:/{model_id}"
        print("Model URI:", model_uri)

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.debug("Model registered: %s version %s", result.name, result.version)
        return result

    except Exception as e:
        logger.error("Error registering model: %s", e)
        raise


def main():
    try:
        model_info = load_model_info("reports/model_info.json")

        run_id = model_info["run_id"]
        model_id = model_info["model_id"]
        model_path = model_info["model_path"]

        print("\nModel Information")
        print("-----------------------------")
        print("Run ID    :", run_id)
        print("Model ID  :", model_id)
        print("Model Path:", model_path)

        model_name = "Sentiment_Analysis_BoW_LR"

        # Register model
        # First Model will be version 1, Second Model is version 2, and so on...
        result = register_model(model_id, model_name)
        registered_version = result.version

        client = MlflowClient()

        # ====================================================
        # Registered Model Description
        # ====================================================

        client.update_registered_model(
            name=model_name,
            description=(
                "Sentiment Analysis model using "
                "Bag of Words (BoW) for text feature extraction "
                "and Logistic Regression for sentiment classification."
            )
        )

        # ====================================================
        # Registered Model Tags
        # ====================================================

        client.set_registered_model_tag(
            name=model_name,
            key="task",
            value="Sentiment Analysis"
        )

        client.set_registered_model_tag(
            name=model_name,
            key="feature_extraction",
            value="Bag of Words"
        )

        client.set_registered_model_tag(
            name=model_name,
            key="algorithm",
            value="Logistic Regression"
        )

        # ====================================================
        # Model Version Description
        # ====================================================

        client.update_model_version(
            name=model_name,
            version=registered_version,
            description=(
                "Sentiment classification model trained using "
                "Bag of Words features and Logistic Regression."
            )
        )

        # ====================================================
        # Model Version Tags
        # ====================================================

        client.set_model_version_tag(
            name=model_name,
            version=registered_version,
            key="algorithm",
            value="LogisticRegression"
        )

        client.set_model_version_tag(
            name=model_name,
            version=registered_version,
            key="vectorizer",
            value="Bag of Words"
        )

        # ====================================================
        # Set Alias for Newly Updated Model Version
        # ====================================================


        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=registered_version
        )


        # ====================================================
        # Final Information
        # ====================================================

        print("\n========================================")
        print("Model successfully registered!")
        print("========================================")

        print(f"Name       : {model_name}")
        print(f"Version    : {registered_version}")
        print(f"Model ID   : {model_id}")
        print(f"Version URI: models:/{model_name}/{registered_version}")
        print(f"Champion   : models:/{model_name}@champion")

        logger.debug("Model registration completed successfully")

    except Exception as e:
        logger.error("Model registration failed: %s", e)
        raise


if __name__ == "__main__":
    main()

