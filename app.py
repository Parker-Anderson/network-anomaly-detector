import joblib
import pandas as pd

from source.preprocessing import load_data
from source.preprocessing import separate_features_and_labels


MODEL_FILE = "models/final_isolation_forest.pkl"
PREPROCESSOR_FILE = "models/final_preprocessor.pkl"
THRESHOLD_FILE = "models/threshold.txt"

TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"


def load_detector():
    print("Loading anomaly detection model...")
    model = joblib.load(MODEL_FILE)

    print("Loading data preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    with open(THRESHOLD_FILE, "r") as file:
        threshold = float(file.read().strip())

    return model, preprocessor, threshold


def main():
    model, preprocessor, threshold = load_detector()

    print("\nNetwork Anomaly Detector")
    print(f"Decision threshold: {threshold:.3f}")

    print("\nLoading UNSW-NB15 test dataset...")
    data = load_data(TESTING_FILE)

    print(f"Dataset contains {len(data):,} records.")

    # select 100 real records
    sample = data.sample(
        n=100,
        random_state=42
    ).copy()

    # keep the labels for evaluation
    actual_labels = sample["label"].values

    # separate features from labels
    X_sample, _ = separate_features_and_labels(sample)

    print("\nPreprocessing network records...")
    X_processed = preprocessor.transform(X_sample)

    print("Calculating anomaly scores...")
    scores = model.decision_function(X_processed)

    # apply the final threshold
    predictions = (scores < threshold).astype(int)

    # demonstration statistics
    correct_predictions = (predictions == actual_labels).sum()
    incorrect_predictions = len(sample) - correct_predictions

    anomalies_detected = (predictions == 1).sum()
    normal_detected = (predictions == 0).sum()

    demo_accuracy = correct_predictions / len(sample)

    # summary
    print("\nDetection Summary")
    print(f"Records analyzed:      {len(sample)}")
    print(f"Correct predictions:   {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Demo accuracy:         {demo_accuracy:.1%}")
    print(f"Anomalies detected:    {anomalies_detected}")
    print(f"Normal traffic:        {normal_detected}")

    # five example results
    print("\nExample Detection Results")

    for i in range(5):
        prediction = (
            "ANOMALY"
            if predictions[i] == 1
            else "NORMAL"
        )

        actual = (
            "ATTACK"
            if actual_labels[i] == 1
            else "NORMAL"
        )

        print(
            f"Record {i + 1}: "
            f"Prediction={prediction}, "
            f"Actual={actual}, "
            f"Score={scores[i]:.4f}"
        )

    print("\nDetection demo complete.")


if __name__ == "__main__":
    main()