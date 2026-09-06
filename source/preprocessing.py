import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state"
]

TARGET_COLUMNS = [
    "id",
    "attack_cat",
    "label"
]


def load_data(file_path):
    """
    Load the UNSW-NB15 dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def separate_features_and_labels(data):
    """
    Separate model features from target information.
    """

    y = data["label"]

    X = data.drop(columns=TARGET_COLUMNS)

    return X, y


def create_preprocessor():
    """
    Create the preprocessing pipeline used by the model.
    """

    numeric_features = [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "swin",
        "stcpb",
        "dtcpb",
        "dwin",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
        "is_sm_ips_ports"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor