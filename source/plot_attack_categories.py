import joblib
import matplotlib.pyplot as plt
import pandas as pd

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

    predictions = (scores < threshold).astype(int)

    # Add predictions back to the original dataset
    results = data.copy()
    results["prediction"] = predictions

    # Only analyze actual attack records
    attacks = results[results["label"] == 1].copy()

    # Calculate detection rate for each attack category
    category_results = (
        attacks
        .groupby("attack_cat")
        .agg(
            total=("label", "count"),
            detected=("prediction", "sum")
        )
        .reset_index()
    )

    category_results["detection_rate"] = (
        category_results["detected"] /
        category_results["total"]
    )

    category_results = category_results.sort_values(
        "detection_rate",
        ascending=True
    )

    print("\nAttack Category Detection")
    print("=========================")

    for _, row in category_results.iterrows():
        print(
            f"{row['attack_cat']}: "
            f"{row['detection_rate']:.2%} "
            f"({int(row['detected'])}/{int(row['total'])})"
        )

    # Create chart
    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        category_results["attack_cat"],
        category_results["detection_rate"]
    )

    plt.xlim(0, 1)
    plt.xlabel("Detection Rate")
    plt.ylabel("Attack Category")
    plt.title(
        "Isolation Forest Detection Rate by Attack Category"
    )

    # Add percentages to bars
    for bar, value in zip(
        bars,
        category_results["detection_rate"]
    ):
        plt.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center"
        )

    plt.tight_layout()

    output_file = "results/attack_category_detection.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nAttack category chart saved to:\n"
        f"{output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()