import joblib
import numpy as np

from preprocessing import load_data
from preprocessing import separate_features_and_labels


TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"
MODEL_FILE = "models/final_isolation_forest.pkl"
PREPROCESSOR_FILE = "models/final_preprocessor.pkl"
THRESHOLD_FILE = "models/threshold.txt"


def main():
    print("Network Anomaly Detector - Monitoring Report")

    print("\nLoading model components...")

    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    with open(THRESHOLD_FILE, "r") as file:
        threshold = float(file.read().strip())

    print("Model loaded successfully.")
    print(f"Decision threshold: {threshold:.4f}")


    print("\nLoading network traffic data...")

    data = load_data(TESTING_FILE)

    print(f"Records loaded: {len(data):,}")

    print("\nData Quality")

    missing_values = data.isnull().sum().sum()

    print(f"Missing values: {missing_values:,}")

    # Check expected categorical values
    expected_protocols = set(
        preprocessor.named_transformers_["categorical"]
        .categories_[0]
    )

    expected_services = set(
        preprocessor.named_transformers_["categorical"]
        .categories_[1]
    )

    expected_states = set(
        preprocessor.named_transformers_["categorical"]
        .categories_[2]
    )

    unexpected_protocols = (
        set(data["proto"].dropna().unique())
        - expected_protocols
    )

    unexpected_services = (
        set(data["service"].dropna().unique())
        - expected_services
    )

    unexpected_states = (
        set(data["state"].dropna().unique())
        - expected_states
    )

    unexpected_categories = (
        len(unexpected_protocols)
        + len(unexpected_services)
        + len(unexpected_states)
    )

    print(f"Unexpected categorical values: {unexpected_categories}")


    print("\nAnalyzing network traffic...")

    X_data, _ = separate_features_and_labels(data)

    X_processed = preprocessor.transform(X_data)

    scores = model.decision_function(X_processed)

    predictions = (scores < threshold).astype(int)

    total_records = len(predictions)

    normal_predictions = np.sum(predictions == 0)
    anomaly_predictions = np.sum(predictions == 1)

    anomaly_rate = anomaly_predictions / total_records

    print("\nPrediction Summary")

    print(f"Records analyzed: {total_records:,}")
    print(f"Normal predictions: {normal_predictions:,}")
    print(f"Anomaly predictions: {anomaly_predictions:,}")
    print(f"Anomaly rate: {anomaly_rate:.2%}")

    print("\nAnomaly Score Statistics")

    print(f"Minimum score: {scores.min():.4f}")
    print(f"Maximum score: {scores.max():.4f}")
    print(f"Average score: {scores.mean():.4f}")
    print(f"Median score: {np.median(scores):.4f}")

    if missing_values > 0:
        status = "WARNING - Missing data detected"

    elif unexpected_categories > 0:
        status = "WARNING - Unexpected categories detected"

    else:
        status = "OPERATIONAL"

    print("\nModel Monitoring Status")

    print(f"Status: {status}")

    print("\nMonitoring completed successfully.")


if __name__ == "__main__":
    main()