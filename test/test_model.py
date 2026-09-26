# Model testing: signature test + performance test

import unittest
import os
import pickle
import mlflow
import pandas as pd
import dagshub

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# MLflow + DagsHub
# mlflow.set_tracking_uri("https://dagshub.com/pranay-majumder/Docker_Pipeline_CI_CD.mlflow")
# dagshub.init(repo_owner="pranay-majumder", repo_name="Docker_Pipeline_CI_CD", mlflow=True)


# MLflow + DagsHub
# Set up DagsHub credentials for MLflow tracking
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


# Load Latest Model (Champion --> Current Model in Production) from MLflow Model Registry
model_name = "Sentiment_Analysis_BoW_LR"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}@champion"
)


# Load BoW Vectorizer
with open("./models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Load holdout data
holdout_data = pd.read_csv(
    "./data/feature_data/test_bow.csv"
)


class TestModel(unittest.TestCase):

    ## Test model signature to ensure that the input and output shapes are as expected.
    def test_model_signature(self):

        input_text = "hi how are you"

        input_data = vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ]
        )

        prediction = model.predict(input_df)

        # Check input shape
        self.assertEqual(
            input_df.shape[1],
            len(vectorizer.get_feature_names_out())
        )

        # Check output shape
        self.assertEqual(
            len(prediction),
            input_df.shape[0]
        )

        self.assertEqual(
            len(prediction.shape),
            1
        )

    ## Test model performance on holdout data to ensure that the model meets the expected performance metrics.
    def test_model_performance(self):

        X_holdout = holdout_data.iloc[:, 0:-1]
        y_holdout = holdout_data.iloc[:, -1]

        y_pred = model.predict(X_holdout)

        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred)

        # Performance thresholds
        self.assertGreaterEqual(accuracy, 0.70)
        self.assertGreaterEqual(precision, 0.70)
        self.assertGreaterEqual(recall, 0.70)
        self.assertGreaterEqual(f1, 0.70)


if __name__ == "__main__":
    unittest.main()