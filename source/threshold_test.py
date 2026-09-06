import joblib
import pandas as pd

from preprocessing import load_data
from preprocessing import separate_features_and_labels

from sklearn.metrics import precision_score, recall_score, f1_score


TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"


def main():
    print("Loading testing dataset...")

    data = load_data(TESTING_FILE)

    X_test, y_test = separate_features_and_labels(data)

    # Load the trained preprocessing pipeline.
    preprocessor = joblib.load("models/preprocessor.pkl")

    # Load the trained Isolation Forest model.
    model = joblib.load("models/isolation_forest.pkl")

    print("Preprocessing testing data...")

    X_test_processed = preprocessor.transform(X_test)

    print("Calculating anomaly scores...")

    scores = model.decision_function(X_test_processed)

    # Thresholds to test.
    thresholds = [
    0.050,
    0.055,
    0.060,
    0.065,
    0.070,
    0.075,
    0.080,
    0.085,
    0.090,
    0.095,
    0.100
]

    results = []

    print("\nTesting thresholds...\n")

    for threshold in thresholds:

        # Lower scores indicate more anomalous traffic.
        predictions = (scores < threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        anomaly_count = predictions.sum()

        results.append({
            "Threshold": threshold,
            "Anomalies Detected": anomaly_count,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()