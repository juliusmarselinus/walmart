
import os
import joblib
import pandas as pd
import numpy as np
from model import load_segmentation_model, MODELS_DIR

def _load_meta():
    path = os.path.join(MODELS_DIR, "model_customer_kmeans.pkl")
    scaler, kmeans, customer_df, feature_cols = load_segmentation_model(path)
    return scaler, kmeans, customer_df, feature_cols

def predict_customer_by_id(customer_id):
    """Return profile dict (if available) and cluster label."""
    _, _, customer_df, _ = _load_meta()
    row = customer_df[customer_df["customer_id"] == customer_id]
    if row.empty:
        return None
    return row.to_dict(orient="records")[0]

def predict_from_features(feature_dict):
    """
    feature_dict: mapping feature_name -> value. Must match feature_columns used in training.
    Returns cluster label and distances to cluster centers (optional).
    """
    scaler, kmeans, _, feature_cols = _load_meta()
    # ensure order of features
    X = []
    for f in feature_cols:
        X.append(feature_dict.get(f, 0))
    X = np.array(X).reshape(1, -1)
    Xs = scaler.transform(X)
    label = int(kmeans.predict(Xs)[0])
    centers = kmeans.transform(Xs).flatten().tolist()  # distances to centers
    return {"cluster": label, "distances": centers}

def predict_from_dataframe(df_customers):
    """
    df_customers: DataFrame with same feature columns as training (customer_id optional).
    Returns df with appended 'cluster' column.
    """
    scaler, kmeans, _, feature_cols = _load_meta()
    # select feature columns if present, else assume df has same order
    missing = [c for c in feature_cols if c not in df_customers.columns]
    if missing:
        # if missing, try to compute simple ones or fill zeros
        for c in missing:
            df_customers[c] = 0
    X = df_customers[feature_cols].fillna(0).values
    Xs = scaler.transform(X)
    labels = kmeans.predict(Xs)
    out = df_customers.copy().reset_index(drop=True)
    out["cluster"] = labels
    return out

