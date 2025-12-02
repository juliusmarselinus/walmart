
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

MODELS_DIR = "models"
DEFAULT_DATA_PATH = "Walmart.csv"

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def build_customer_agg(df):
    """
    Build customer-level aggregate features from transactional df.
    Returns customer_df (index=customer_id) with numeric features.
    """
    if "customer_id" not in df.columns:
        raise ValueError("Dataset harus berisi kolom 'customer_id' untuk segmentation.")

    # Basic aggregates
    agg = df.groupby("customer_id").agg(
        total_qty = ("quantity_sold", "sum") if "quantity_sold" in df.columns else ("product_id", "count"),
        avg_price = ("unit_price", "mean") if "unit_price" in df.columns else ("quantity_sold", "mean"),
        n_transactions = ("transaction_id", "nunique") if "transaction_id" in df.columns else ("product_id", "nunique")
    )

    # optional features
    if "customer_age" in df.columns:
        agg["age_median"] = df.groupby("customer_id")["customer_age"].median()
    if "customer_loyalty_level" in df.columns:
        # convert loyalty level to frequency encoding
        lo_freq = df.groupby("customer_id")["customer_loyalty_level"].agg(lambda x: x.mode().iat[0] if len(x.mode())>0 else np.nan)
        # one-hot / map rare -> numeric by ordinal encoding via category codes
        try:
            agg["loyalty_code"] = lo_freq.astype("category").cat.codes
        except Exception:
            agg["loyalty_code"] = lo_freq.fillna(-1)

    # recency if transaction_date present
    if "transaction_date" in df.columns:
        try:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])
            last_date = df["transaction_date"].max()
            rec = df.groupby("customer_id")["transaction_date"].max().apply(lambda d: (last_date - d).days)
            agg["recency_days"] = rec
        except Exception:
            pass

    # fillna and ensure numeric
    agg = agg.fillna(0)
    # if any columns non-numeric (shouldn't), convert
    for c in agg.columns:
        if not pd.api.types.is_numeric_dtype(agg[c]):
            try:
                agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0)
            except Exception:
                agg[c] = 0
    return agg

def train_customer_segmentation(data_path=DEFAULT_DATA_PATH, n_clusters=4, random_state=42):
    """
    Train KMeans segmentation and save meta to models/model_customer_kmeans.pkl.
    Returns metadata dict.
    """
    ensure_dirs()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)
    customer_df = build_customer_agg(df)

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(customer_df.values)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    customer_df["cluster"] = labels

    # Prepare customer profile list for quick lookup
    customer_profile = customer_df.reset_index().rename(columns={"index": "customer_id"})
    meta = {
        "scaler": scaler,
        "kmeans": kmeans,
        "customer_profile": customer_profile.to_dict(orient="records"),
        "feature_columns": customer_df.columns.drop("cluster").tolist()
    }

    out_path = os.path.join(MODELS_DIR, "model_customer_kmeans.pkl")
    joblib.dump(meta, out_path)
    summary = {
        "path": out_path,
        "n_customers": customer_df.shape[0],
        "n_clusters": int(n_clusters)
    }
    return summary

def load_segmentation_model(path=os.path.join(MODELS_DIR, "model_customer_kmeans.pkl")):
    if not os.path.exists(path):
        raise FileNotFoundError("Model segmentation tidak ditemukan. Jalankan train_customer_segmentation dulu.")
    meta = joblib.load(path)
    # convert customer_profile back to DataFrame for convenience
    customer_df = pd.DataFrame(meta.get("customer_profile", []))
    return meta["scaler"], meta["kmeans"], customer_df, meta.get("feature_columns", [])

if __name__ == "__main__":
    print("Contoh: melatih segmentation dengan n_clusters=4 dari Walmart.csv")
    print(train_customer_segmentation())
