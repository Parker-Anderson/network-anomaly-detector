import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import load_data
from preprocessing import separate_features_and_labels
from preprocessing import create_preprocessor


TRAINING_FILE = "data/raw/UNSW_NB15_testing-set.csv"


def main():

    print("Loading training dataset...")

    data = load_data(TRAINING_FILE)

    # Use only normal traffic.
    normal_data = data[data["label"] == 0].copy()

    # Separate normal data into training and validation portions.
    normal_train = normal_data.sample(
        frac=0.80,
        random_state=42
    )

    normal_validation = normal_data.drop(
        normal_train.index
    )

    # All attacks are included in validation.
    attack_validation = data[data["label"] == 1].copy()

    validation_data = pd.concat(
        [normal_validation, attack_validation],
        ignore_index=True
    )

    print(f"Normal training records: {len(normal_train)}")
    print(f"Normal validation records: {len(normal_validation)}")
    print(f"Attack validation records: {len(attack_validation)}")
    print(f"Total validation records: {len(validation_data)}")

    # Prepare training data.
    X_normal_train, _ = separate_features_and_labels(
        normal_train
    )

    # Prepare validation data.
    X_validation, y_validation = separate_features_and_labels(
        validation_data
    )

    # Create and fit preprocessing pipeline using ONLY normal training data.
    preprocessor = create_preprocessor()

    print("\nPreprocessing data...")

    X_normal_train_processed = preprocessor.fit_transform(
        X_normal_train
    )

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    print(
        f"Processed training shape: "
        f"{X_normal_train_processed.shape}"
    )

    print(
        f"Processed validation shape: "
        f"{X_validation_processed.shape}"
    )

    contamination_values = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.15,
        0.20
    ]

    thresholds = [
        0.025,
        0.050,
        0.075,
        0.100,
        0.125,
        0.150
    ]

    print("\nTesting contamination values...")

    best_result = None

    for contamination in contamination_values:

        print(
            f"\nTraining model with "
            f"contamination={contamination}"
        )

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_normal_train_processed)

        scores = model.decision_function(
            X_validation_processed
        )

        for threshold in thresholds:

            predictions = (
                scores < threshold
            ).astype(int)

            precision = precision_score(
                y_validation,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_validation,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_validation,
                predictions,
                zero_division=0
            )

            normal_mask = y_validation == 0

            false_positive_rate = np.mean(
                predictions[normal_mask] == 1
            )

            result = {
                "contamination": contamination,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": false_positive_rate
            }

            if (
                best_result is None
                or f1 > best_result["f1"]
            ):
                best_result = result

            print(
                f"contamination={contamination:<5} "
                f"threshold={threshold:<6.3f} "
                f"precision={precision:.4f} "
                f"recall={recall:.4f} "
                f"F1={f1:.4f} "
                f"FPR={false_positive_rate:.4f}"
            )

    print("\n")

    print("Best Validation Result")

    print(
        f"Contamination:      "
        f"{best_result['contamination']}"
    )

    print(
        f"Threshold:          "
        f"{best_result['threshold']:.3f}"
    )

    print(
        f"Precision:          "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"False Positive Rate:"
        f" {best_result['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()