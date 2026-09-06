import joblib
import pandas as pd
import numpy as np

from preprocessing import load_data
from preprocessing import separate_features_and_labels


TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"

MODEL_FILE = "models/final_isolation_forest.pkl"
PREPROCESSOR_FILE = "models/final_preprocessor.pkl"


def main():

    print("Loading final model...")
    model = joblib.load(MODEL_FILE)

    print("Loading final preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    print("\nLoading test dataset...")

    data = load_data(TESTING_FILE)

    X_test, y_test = separate_features_and_labels(data)

    attack_categories = data["attack_cat"]

    print(f"Dataset shape: {data.shape}")

    print("\nPreprocessing test data...")

    X_test_processed = preprocessor.transform(X_test)

    print("Calculating anomaly scores...")

    scores = model.decision_function(X_test_processed)

    results = pd.DataFrame({
        "attack_cat": attack_categories,
        "label": y_test,
        "score": scores
    })

    print("\nAttack Category Analysis")
    print("========================")

    attack_data = results[results["label"] == 1]

    categories = attack_data["attack_cat"].unique()

    print(
        f"{'Attack Category':<20}"
        f"{'Count':<10}"
        f"{'Mean Score':<15}"
        f"{'Median Score':<15}"
        f"{'Detected @ 0.100':<20}"
    )

    print("-" * 80)

    for category in sorted(categories):

        category_data = attack_data[
            attack_data["attack_cat"] == category
        ]

        category_scores = category_data["score"]

        detected = np.sum(
            category_scores < 0.100
        )

        detection_rate = (
            detected / len(category_scores)
        ) * 100

        print(
            f"{category:<20}"
            f"{len(category_scores):<10}"
            f"{category_scores.mean():<15.6f}"
            f"{category_scores.median():<15.6f}"
            f"{detection_rate:>8.2f}%"
        )


if __name__ == "__main__":
    main()