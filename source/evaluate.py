import joblib

from preprocessing import load_data
from preprocessing import separate_features_and_labels

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)



TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"


def main():
    print("Loading testing dataset...")

    data = load_data(TESTING_FILE)

    print(f"Testing dataset shape: {data.shape}")

    # Separate features and labels.
    X_test, y_test = separate_features_and_labels(data)

    # Load the trained preprocessing pipeline.
    preprocessor = joblib.load("models/preprocessor.pkl")

    # Load the trained Isolation Forest model.
    model = joblib.load("models/isolation_forest.pkl")

    print("Preprocessing testing data...")

    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed testing shape: {X_test_processed.shape}")

    print("Running anomaly detection...")

    scores = model.decision_function(X_test_processed)

    predictions = model.predict(X_test_processed)

    print("\nAnomaly score statistics:")
    print(f"Minimum score: {scores.min():.4f}")
    print(f"Maximum score: {scores.max():.4f}")
    print(f"Average score: {scores.mean():.4f}")

    # Convert Isolation Forest predictions:
    #  1  = normal
    # -1  = anomaly
    #
    # Our dataset uses:
    #  0  = normal
    #  1  = attack
    predictions_binary = (predictions == -1).astype(int)

    print("\nDetection results:")

    normal_count = (predictions_binary == 0).sum()
    anomaly_count = (predictions_binary == 1).sum()

    print(f"Normal predictions: {normal_count}")
    print(f"Anomaly predictions: {anomaly_count}")
    print(f"Total predictions: {len(predictions_binary)}")

    # Calculate accuracy.
    accuracy = accuracy_score(y_test, predictions_binary)

    print("\nModel Accuracy:")
    print(f"{accuracy:.4f}")

    # Confusion matrix.
    matrix = confusion_matrix(y_test, predictions_binary)

    print("\nConfusion Matrix:")
    print(matrix)

    # Detailed classification report.
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions_binary,
            target_names=["Normal", "Attack"]
        )
    )


if __name__ == "__main__":
    main()