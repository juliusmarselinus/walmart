import streamlit as st
import pandas as pd
import os
from model import train_customer_segmentation, load_segmentation_model, MODELS_DIR
from prediction import predict_from_dataframe
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation - KMeans", layout="wide")
st.title("Customer Segmentation (KMeans) — Walmart")


DATA_UP_PATH = "Walmart.csv"
MODEL_PATH = os.path.join(MODELS_DIR, "model_customer_kmeans.pkl")

# Sidebar — upload + training
st.sidebar.header("Data & Training")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (transaction-level) untuk train segmentation", type=["csv"])
if uploaded:
    with open(DATA_UP_PATH, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Dataset disimpan ke {DATA_UP_PATH}")

n_clusters = st.sidebar.slider("Jumlah cluster (K)", min_value=2, max_value=10, value=4)

if st.sidebar.button("Train Segmentation"):
    try:
        from model import train_customer_segmentation
        with st.spinner("Melatih segmentation (KMeans)..."):
            res = train_customer_segmentation(data_path=DATA_UP_PATH, n_clusters=n_clusters)
        st.sidebar.success("Training selesai")
        st.sidebar.json(res)
    except Exception as e:
        st.sidebar.error(f"Training gagal: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Model status:")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("Model segmentation tersedia.")
else:
    st.sidebar.warning("Model belum tersedia. Upload dataset & Train.")

# Main area
if os.path.exists(MODEL_PATH):
    scaler, kmeans, customer_df, feature_cols = load_segmentation_model(MODEL_PATH)

    st.header("Cluster Overview")
    st.write(f"Found {customer_df.shape[0]} customers; features used: {feature_cols}")

    # Cluster counts
    st.subheader("Cluster counts")
    counts = customer_df["cluster"].value_counts().sort_index()
    st.bar_chart(counts)

    # PCA visualization
    st.subheader("2D visualization (PCA)")
    try:
        pca = PCA(n_components=2)
        X = scaler.transform(customer_df[feature_cols].fillna(0).values)
        X2 = pca.fit_transform(X)
        vis_df = pd.DataFrame(X2, columns=["pc1", "pc2"])
        vis_df["cluster"] = customer_df["cluster"].values

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=vis_df, x="pc1", y="pc2", hue="cluster", palette="tab10", ax=ax)
        ax.set_title("PCA projection colored by cluster")
        st.pyplot(fig)
    except Exception as e:
        st.write("Visualisasi PCA gagal:", e)

   
