from preprocessing import load_data
from preprocessing import separate_features_and_labels
from preprocessing import create_preprocessor
from model import create_model
import joblib


TRAINING_FILE = "data/raw/UNSW_NB15_testing-set.csv"


def main():
    print("Loading dataset...")

    data = load_data(TRAINING_FILE)

    print(f"Dataset shape: {data.shape}")

    # Keep only normal network traffic for training.
    normal_data = data[data["label"] == 0].copy()

    print(f"Normal records used for training: {len(normal_data)}")

    # Separate features from labels.
    X_normal, y_normal = separate_features_and_labels(normal_data)

    # Create preprocessing pipeline.
    preprocessor = create_preprocessor()

    print("Preprocessing normal traffic...")

    X_normal_processed = preprocessor.fit_transform(X_normal)

    print(f"Processed training shape: {X_normal_processed.shape}")

       # Create Isolation Forest model.
    model = create_model()

    print("Training Isolation Forest...")

    model.fit(X_normal_processed)

    print("Training completed successfully!")

    # Save the trained model.
    joblib.dump(model, "models/isolation_forest.pkl")

    # Save the preprocessing pipeline.
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Model saved to models/isolation_forest.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")

if __name__ == "__main__":
    main()