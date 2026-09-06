import joblib
import numpy as np

from preprocessing import load_data
from preprocessing import create_preprocessor
from preprocessing import separate_features_and_labels

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


TRAINING_FILE = "data/raw/UNSW_NB15_testing-set.csv"


def main():
    print("Loading training dataset...")

    data = load_data(TRAINING_FILE)

    print(f"Dataset shape: {data.shape}")

    # Separate normal and attack traffic.
    normal_data = data[data["label"] == 0].copy()
    attack_data = data[data["label"] == 1].copy()

    print(f"Normal records: {len(normal_data)}")
    print(f"Attack records: {len(attack_data)}")

    # Split the normal traffic into training and validation data.
    normal_train, normal_validation = train_test_split(
        normal_data,
        test_size=0.20,
        random_state=42
    )

    print(f"Normal training records: {len(normal_train)}")
    print(f"Normal validation records: {len(normal_validation)}")

    # Build validation data using all validation normal traffic
    # plus all available attack traffic.
    validation_data = np.concatenate([
        normal_validation.index.to_numpy(),
        attack_data.index.to_numpy()
    ])

    validation_data = data.loc[validation_data]

    print(f"Validation records: {len(validation_data)}")

    # Prepare training features.
    X_train, _ = separate_features_and_labels(normal_train)

    # Prepare validation features and labels.
    X_validation, y_validation = separate_features_and_labels(
        validation_data
    )

    # Create preprocessing pipeline.
    preprocessor = create_preprocessor()

    print("\nPreprocessing training data...")

    X_train_processed = preprocessor.fit_transform(X_train)

    print(f"Processed training shape: {X_train_processed.shape}")

    print("\nPreprocessing validation data...")

    X_validation_processed = preprocessor.transform(X_validation)

    print(
        f"Processed validation shape: "
        f"{X_validation_processed.shape}"
    )

    # Create a fresh Isolation Forest model.
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")

    model.fit(X_train_processed)

    print("Training completed!")

    # Generate anomaly scores for validation data.
    print("\nCalculating validation anomaly scores...")

    scores = model.decision_function(X_validation_processed)

    # Test thresholds.
    thresholds = [
        0.100,
        0.105,
        0.110,
        0.115,
        0.120,
        0.125,
        0.130,
        0.135,
        0.140,
        0.145,
        0.150
    ]

    print("\nValidation threshold results:")
    print()

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:

        predictions = (scores < threshold).astype(int)

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

        anomaly_count = predictions.sum()

        print(
            f"Threshold: {threshold:.3f} | "
            f"Anomalies: {anomaly_count} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\nBest validation threshold:")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"F1 Score: {best_f1:.4f}")

    # Save the validation-trained model and preprocessor.
    joblib.dump(model, "models/isolation_forest_validation.pkl")
    joblib.dump(preprocessor, "models/preprocessor_validation.pkl")

    # Save the selected threshold.
    with open("models/threshold.txt", "w") as file:
        file.write(str(best_threshold))

    print("\nValidation model saved.")
    print("Selected threshold saved to models/threshold.txt")


if __name__ == "__main__":
    main()