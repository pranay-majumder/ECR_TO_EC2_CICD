# Model Building
# Here I am Using "Logistric Regression + BOW"
# Best Model (Logistic Regression) + Best Hyper Prameter Obtain from Experiment Tracking.

import os
import pickle
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Logging
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def train_model(X_train, y_train):
    try:
        model = LogisticRegression(C=1, solver="liblinear", penalty="l2")
        model.fit(X_train, y_train)
        logger.debug("Model training completed")
        return model
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)
            
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise


def main():
    try:
        train_data = load_data("./data/feature_data/train_bow.csv")
        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values

        model = train_model(X_train, y_train)
        save_model(model, "models/model.pkl")

    except Exception as e:
        logger.error("Model building failed: %s", e)
        raise


if __name__ == "__main__":
    main()