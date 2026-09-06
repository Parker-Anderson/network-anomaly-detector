import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import load_data
from preprocessing import separate_features_and_labels


TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"

MODEL_FILE = "models/final_isolation_forest.pkl"
PREPROCESSOR_FILE = "models/final_preprocessor.pkl"
THRESHOLD_FILE = "models/threshold.txt"


def main():

    print("Loading final model...")
    model = joblib.load(MODEL_FILE)

    print("Loading final preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    # Load the selected threshold.
    with open(THRESHOLD_FILE, "r") as file:
        threshold = float(file.read().strip())

    print(f"Using threshold: {threshold:.3f}")

    print("\nLoading untouched test dataset...")

    data = load_data(TESTING_FILE)

    print(f"Test dataset shape: {data.shape}")

    # Separate features from actual labels.
    X_test, y_test = separate_features_and_labels(data)

    print(f"Test feature shape: {X_test.shape}")
    print(f"Test label shape: {y_test.shape}")

    print("\nPreprocessing test data...")

    # IMPORTANT:
    # Transform only. Do not fit the preprocessor on test data.
    X_test_processed = preprocessor.transform(X_test)

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    print("\nCalculating anomaly scores...")

    scores = model.decision_function(X_test_processed)

    # Lower Isolation Forest scores indicate more anomalous data.
    predictions = (scores < threshold).astype(int)

    print("\nFinal Evaluation Results")

    accuracy = accuracy_score(y_test, predictions)
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

    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix")

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Attack"]
        )
    )

    print("\nPrediction Counts")

    print(
        f"Normal predictions: "
        f"{np.sum(predictions == 0)}"
    )

    print(
        f"Anomaly predictions: "
        f"{np.sum(predictions == 1)}"
    )


if __name__ == "__main__":
    main()