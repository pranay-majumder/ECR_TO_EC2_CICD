import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["tweet_id"], inplace=True)
        df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        df["sentiment"] = df["sentiment"].replace({"happiness": 1, "sadness": 0})
        logger.debug("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


def save_data(train_data, test_data, data_path: str) -> None:
    try:
        raw_path = os.path.join(data_path, "raw_data")
        
        ## Create Directory
        os.makedirs(raw_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_path, "test.csv"), index=False)
        logger.debug("Train and test data saved to %s", raw_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(
            "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        )
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, "./data")

    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        raise


if __name__ == "__main__":
    main()