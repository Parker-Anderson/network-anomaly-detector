from sklearn.ensemble import IsolationForest
import joblib


def create_model():


    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    return model