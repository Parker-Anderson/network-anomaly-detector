import joblib

from sklearn.ensemble import IsolationForest

from preprocessing import load_data
from preprocessing import separate_features_and_labels
from preprocessing import create_preprocessor


TRAINING_FILE = "data/raw/UNSW_NB15_testing-set.csv"

CONTAMINATION = 0.01
THRESHOLD = 0.125


def main():

    print("Loading complete training dataset...")

    data = load_data(TRAINING_FILE)

    print(f"Dataset shape: {data.shape}")

    # Use only normal traffic for final model training.
    normal_data = data[data["label"] == 0].copy()

    print(
        f"Normal records used for final training: "
        f"{len(normal_data)}"
    )

    # Separate features from labels.
    X_normal, _ = separate_features_and_labels(
        normal_data
    )

    # Create preprocessing pipeline.
    preprocessor = create_preprocessor()

    print("\nPreprocessing training data...")

    X_normal_processed = preprocessor.fit_transform(
        X_normal
    )

    print(
        f"Processed training shape: "
        f"{X_normal_processed.shape}"
    )

    # Create final Isolation Forest.
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining final Isolation Forest...")
    print(f"Contamination: {CONTAMINATION}")
    print(f"Decision threshold: {THRESHOLD}")

    model.fit(X_normal_processed)

    print("\nFinal model training completed!")

    # Save final model.
    joblib.dump(
        model,
        "models/final_isolation_forest.pkl"
    )

    # Save final preprocessor.
    joblib.dump(
        preprocessor,
        "models/final_preprocessor.pkl"
    )

    # Save threshold.
    with open(
        "models/threshold.txt",
        "w"
    ) as file:
        file.write(str(THRESHOLD))

    print("\nFinal model saved to:")
    print("models/final_isolation_forest.pkl")

    print("\nFinal preprocessor saved to:")
    print("models/final_preprocessor.pkl")

    print("\nFinal threshold saved to:")
    print("models/threshold.txt")


if __name__ == "__main__":
    main()