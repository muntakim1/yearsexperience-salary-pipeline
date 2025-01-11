import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
import yaml
import logging
import mlflow
import mlflow.sklearn
import joblib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

mlflow.set_tracking_uri("http://localhost:5500")
n_jobs = params["training"]["n_jobs"]
random_state = params["training"]["random_state"]
cv = params["training"]["cv"]


def evaluate():
    "Train the model with scaled data"
    logging.info("Training the model")

    try:
        test = pd.read_csv("data/processed/test_scaled.csv")
        logging.info("Test data loaded from data/processed/test_scaled.csv")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return

    X_test = test.drop("Salary", axis=1)
    y_test = test["Salary"].values

    # Load the saved scaler
    scaler = joblib.load("models/scaler.pkl")
    logging.info("Scaler loaded from scaler.pkl")

    X_test_scaled = scaler.transform(X_test)

    model = joblib.load("models/model.pkl")

    # Model evaluation
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"Test Mean Absolute Error: {mae}")

    # Log metrics and model to MLFlow
    try:
        with mlflow.start_run():
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_metric("test mae", mae)
            mlflow.sklearn.log_model(model, "model")
            logging.info("Model logged in MLFlow")
    except Exception as e:
        logging.error(f"Error logging model with MLFlow: {e}")

    # Save metrics to a JSON file
    metrics = {"Test mae": mae}
    with open("metrics/metrics.yaml", "w") as file:
        yaml.safe_dump(metrics, file)

    logging.info("Training completed")


if __name__ == "__main__":
    evaluate()
