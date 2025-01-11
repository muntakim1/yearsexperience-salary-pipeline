import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import joblib
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["preprocessing"]["test_size"]
random_state = params["preprocessing"]["random_state"]


def preprocess():
    "Preprocess the data, scale it, and save the scaler"
    logging.info("Preprocessing the data")

    # Load the dataset
    data = pd.read_csv("data/raw/Salary_Data.csv")
    logging.info("Data loaded from data/raw/Salary_Data.csv")

    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    logging.info("Data split into train and test")

    # Perform standard scaling on numeric columns
    scaler = StandardScaler()

    # Assuming 'YearsExperience' is the feature and 'Salary' is the target
    X_train = train[["YearsExperience"]]
    y_train = train["Salary"]
    X_test = test[["YearsExperience"]]
    y_test = test["Salary"]

    # Scaling the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to a file for later use
    joblib.dump(scaler, "models/scaler.pkl")
    logging.info("Scaler saved to scaler.pkl")

    # Create DataFrames with the scaled data
    train_scaled = pd.DataFrame(X_train_scaled, columns=["YearsExperience"])
    test_scaled = pd.DataFrame(X_test_scaled, columns=["YearsExperience"])

    # Adding target columns back to the data
    train_scaled["Salary"] = y_train.values
    test_scaled["Salary"] = y_test.values

    # Save the processed data
    train_scaled.to_csv("data/processed/train_scaled.csv", index=False)
    logging.info("Train data with scaling saved to data/processed/train_scaled.csv")

    test_scaled.to_csv("data/processed/test_scaled.csv", index=False)
    logging.info("Test data with scaling saved to data/processed/test_scaled.csv")

    logging.info("Preprocessing completed")


if __name__ == "__main__":
    preprocess()
