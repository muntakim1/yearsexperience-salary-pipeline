import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
import yaml
import logging
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

mlflow.set_tracking_uri("http://localhost:5500")
n_jobs = params["training"]["n_jobs"]
random_state = params["training"]["random_state"]
cv = params["training"]["cv"]


def train():
    "Train the model with scaled data"
    logging.info("Training the model")

    try:
        # Load train and test data
        train = pd.read_csv("data/processed/train_scaled.csv")
        logging.info("Train data loaded from data/processed/train_scaled.csv")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return

    X_train = train.drop("Salary", axis=1)
    y_train = train["Salary"].values

    # Load the saved scaler
    scaler = joblib.load("models/scaler.pkl")
    logging.info("Scaler loaded from scaler.pkl")

    # Ensure the scaler is applied to both train and test data
    X_train_scaled = scaler.transform(X_train)

    model = LinearRegression(n_jobs=n_jobs)

    # Cross-validation setup
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=kfold, scoring=scorer
    )
    logging.info(f"Mean CV Score: {cv_scores.mean()}")
    logging.info(f"Std CV Score: {cv_scores.std()}")

    # Train the model with the scaled data
    model.fit(X_train_scaled, y_train)
    logging.info("Model trained")

    # Model evaluation
    predictions = model.predict(X_train_scaled)
    mae = mean_absolute_error(y_train, predictions)
    logging.info(f"Training Mean Absolute Error: {mae}")
    joblib.dump(model, "models/model.pkl")
    # Log metrics and model to MLFlow
    try:
        with mlflow.start_run():
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("cv", cv)
            mlflow.log_metric("traning mae", mae)
            plt.figure(figsize=(8, 8))
            plt.scatter(y_train, predictions, color="blue", label="Predictions")
            plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color="red", linestyle="--", label="Ideal Line")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs Actual")
            plt.legend()

            # Save the plot
            plot_path = "predicted_vs_actual.png"
            plt.savefig(plot_path)
            plt.close()

            # Log the plot as an artifact
            mlflow.log_artifact(plot_path)

            mlflow.sklearn.log_model(model, "model")
            logging.info("Model logged in MLFlow")
    except Exception as e:
        logging.error(f"Error logging model with MLFlow: {e}")

    logging.info("Training completed")


if __name__ == "__main__":
    train()
