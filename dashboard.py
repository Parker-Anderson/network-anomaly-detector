import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from source.preprocessing import load_data
from source.preprocessing import separate_features_and_labels


MODEL_FILE = "models/final_isolation_forest.pkl"
PREPROCESSOR_FILE = "models/final_preprocessor.pkl"
THRESHOLD_FILE = "models/threshold.txt"

TESTING_FILE = "data/raw/UNSW_NB15_training-set.csv"


st.set_page_config(
    page_title="Network Anomaly Detector",
    layout="wide"
)

# load the model and data


@st.cache_resource
def load_detector():

    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    with open(THRESHOLD_FILE, "r") as file:
        threshold = float(file.read().strip())

    return model, preprocessor, threshold


@st.cache_data
def load_network_data():

    return load_data(TESTING_FILE)


model, preprocessor, threshold = load_detector()
data = load_network_data()

# Run predictions


X, y = separate_features_and_labels(data)

X_processed = preprocessor.transform(X)

scores = model.decision_function(X_processed)

predictions = (scores < threshold).astype(int)


# Calculate metrics

correct = (predictions == y.values).sum()
accuracy = correct / len(y)

normal_count = (predictions == 0).sum()
anomaly_count = (predictions == 1).sum()

matrix = confusion_matrix(y, predictions)

tn, fp, fn, tp = matrix.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


st.title("Parker Anderson \n" \
" Isolation Forest Network Anomaly Detector")


st.subheader("Interactive Network Record Tool")

record_number = st.number_input(
    "Enter a record number:",
    min_value=1,
    max_value=len(data),
    value=1
)

index = record_number - 1

record = data.iloc[index]

prediction = predictions[index]
score = scores[index]
actual = y.iloc[index]

if prediction == 1:
    prediction_text = "ANOMALY"
else:
    prediction_text = "NORMAL"

if actual == 1:
    actual_text = "ATTACK"
else:
    actual_text = "NORMAL"


col1, col2, col3 = st.columns(3)

col1.metric(
    "Prediction",
    prediction_text
)

col2.metric(
    "Anomaly Score",
    f"{score:.4f}"
)

col3.metric(
    "Actual Classification",
    actual_text
)


st.write(" Network Record")

st.dataframe(
    record.iloc[:10].to_frame().T,
    use_container_width=True,
    hide_index=True
)


# metrics

st.subheader("Detection Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)


# data summary

st.subheader("Network Traffic Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Records Analyzed",
    f"{len(data):,}"
)

col2.metric(
    "Anomalies Detected",
    f"{anomaly_count:,}"
)

col3.metric(
    "Normal Traffic",
    f"{normal_count:,}"
)


st.write(
    f"Decision threshold: **{threshold:.3f}**"
)


# visualization 1 - confusion matrix

st.subheader("1. Confusion Matrix")

fig, ax = plt.subplots()

display_matrix = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Normal", "Attack"]
)

display_matrix.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Network Traffic Classification")

st.pyplot(fig)

# visualization 2 -prediction distribution


st.subheader("2. Detection Results")

prediction_counts = pd.Series(
    predictions
).map({
    0: "Normal",
    1: "Anomaly"
}).value_counts()

fig, ax = plt.subplots()

prediction_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Normal vs. Anomalous Predictions")
ax.set_xlabel("Classification")
ax.set_ylabel("Number of Records")

st.pyplot(fig)


# visualization 3 - anomaly score distribution

st.subheader("3. Anomaly Score Distribution")

normal_scores = scores[y.values == 0]
attack_scores = scores[y.values == 1]

fig, ax = plt.subplots()

ax.hist(
    normal_scores,
    bins=40,
    alpha=0.6,
    label="Normal"
)

ax.hist(
    attack_scores,
    bins=40,
    alpha=0.6,
    label="Attack"
)

ax.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold ({threshold:.3f})"
)

ax.set_title("Distribution of Isolation Forest Anomaly Scores")
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Number of Records")
ax.legend()

st.pyplot(fig)




# attack categories


st.subheader("Attack Category Detection")

category_data = data.copy()

category_data["prediction"] = predictions

attack_data = category_data[
    category_data["label"] == 1
].copy()

category_detection = (
    attack_data
    .groupby("attack_cat")["prediction"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

fig, ax = plt.subplots()

category_detection.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Attack Detection Rate by Category")
ax.set_xlabel("Attack Category")
ax.set_ylabel("Detection Rate (%)")

ax.set_ylim(0, 100)

st.pyplot(fig)

st.subheader("Maintenance Monitoring")

st.write(
    "This section monitors data quality and model behavior "
    "to help identify potential maintenance or retraining needs."
)

# Data quality checks
missing_values = data.isnull().sum().sum()

expected_protocols = set(
    preprocessor.named_transformers_["categorical"].categories_[0]
)

expected_services = set(
    preprocessor.named_transformers_["categorical"].categories_[1]
)

expected_states = set(
    preprocessor.named_transformers_["categorical"].categories_[2]
)

unexpected_protocols = (
    set(data["proto"].dropna().unique())
    - expected_protocols
)

unexpected_services = (
    set(data["service"].dropna().unique())
    - expected_services
)

unexpected_states = (
    set(data["state"].dropna().unique())
    - expected_states
)

unexpected_categories = (
    len(unexpected_protocols)
    + len(unexpected_services)
    + len(unexpected_states)
)

# Monitoring statistics
total_records = len(predictions)

normal_predictions = (predictions == 0).sum()
anomaly_predictions = (predictions == 1).sum()

anomaly_rate = anomaly_predictions / total_records

average_score = scores.mean()
median_score = np.median(scores)

# Model status
if missing_values > 0:
    model_status = "WARNING - Missing Data"

elif unexpected_categories > 0:
    model_status = "WARNING - Unexpected Categories"

else:
    model_status = "OPERATIONAL"

# Display monitoring metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Records Monitored",
    f"{total_records:,}"
)

col2.metric(
    "Anomaly Rate",
    f"{anomaly_rate:.2%}"
)

col3.metric(
    "Average Anomaly Score",
    f"{average_score:.4f}"
)

col4.metric(
    "Decision Threshold",
    f"{threshold:.4f}"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Missing Values",
    f"{missing_values:,}"
)

col2.metric(
    "Unexpected Categories",
    f"{unexpected_categories:,}"
)

col3.metric(
    "Model Status",
    model_status
)

st.caption(
    "Parker Anderson | C964 | Network Anomaly Detector"
)