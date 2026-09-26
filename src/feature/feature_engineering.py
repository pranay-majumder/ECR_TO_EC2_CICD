# Feature Engineering

import os
import pickle
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Logging
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("feature_engineering_errors.log")
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


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path).fillna("")
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def apply_bow(train_data, test_data, max_features):
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = vectorizer.fit_transform(train_data["content"])
        X_test = vectorizer.transform(test_data["content"])

        train_df = pd.DataFrame(X_train.toarray())
        train_df["label"] = train_data["sentiment"].values

        test_df = pd.DataFrame(X_test.toarray())
        test_df["label"] = test_data["sentiment"].values

        ## It is Required During Prediction (vectorizer.pkl)
        os.makedirs("models", exist_ok=True)
        with open("models/vectorizer.pkl", "wb") as file:
            pickle.dump(vectorizer, file)

        logger.debug("Bag of Words transformation completed")
        return train_df, test_df

    except Exception as e:
        logger.error("Error during BoW transformation: %s", e)
        raise


def save_data(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data("./data/processed_data/train_processed.csv")
        test_data = load_data("./data/processed_data/test_processed.csv")

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, "./data/feature_data/train_bow.csv")
        save_data(test_df, "./data/feature_data/test_bow.csv")

    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()