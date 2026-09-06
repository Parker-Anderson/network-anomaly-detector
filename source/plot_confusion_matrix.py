import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    with open(THRESHOLD_FILE, "r") as file:
        threshold = float(file.read().strip())

    print(f"Using threshold: {threshold:.3f}")

    print("\nLoading test dataset...")

    data = load_data(TESTING_FILE)

    X_test, y_test = separate_features_and_labels(data)

    print("Preprocessing test data...")

    X_test_processed = preprocessor.transform(X_test)

    print("Calculating predictions...")

    scores = model.decision_function(X_test_processed)

    predictions = (scores < threshold).astype(int)

    # Create confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Create visualization.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Attack"]
    )

    display.plot()

    plt.title(
        "UNSW-NB15 Network Anomaly Detection\n"
        f"Isolation Forest (Threshold = {threshold:.3f})"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    # Save the figure.
    output_file = "results/confusion_matrix.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nConfusion matrix saved to:")
    print(output_file)

    plt.show()


if __name__ == "__main__":
    main()