import joblib
import matplotlib.pyplot as plt
import numpy as np

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

    print("Calculating anomaly scores...")
    scores = model.decision_function(X_test_processed)

    normal_scores = scores[y_test == 0]
    attack_scores = scores[y_test == 1]

    print("\nScore Statistics")
    print("================")
    print(f"Normal records: {len(normal_scores)}")
    print(f"Attack records: {len(attack_scores)}")

    print(f"\nNormal mean score: {np.mean(normal_scores):.4f}")
    print(f"Attack mean score: {np.mean(attack_scores):.4f}")

    print(f"\nNormal median score: {np.median(normal_scores):.4f}")
    print(f"Attack median score: {np.median(attack_scores):.4f}")

    # Create histogram
    plt.figure(figsize=(10, 6))

    plt.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label="Normal Traffic"
    )

    plt.hist(
        attack_scores,
        bins=50,
        alpha=0.6,
        label="Attack Traffic"
    )

    # Decision threshold
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Decision Threshold ({threshold:.3f})"
    )

    plt.xlabel("Isolation Forest Decision Score")
    plt.ylabel("Number of Records")
    plt.title(
        "Isolation Forest Anomaly Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output_file = "results/anomaly_score_distribution.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nAnomaly score distribution saved to:\n"
        f"{output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()