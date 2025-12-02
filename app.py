# app.py
import streamlit as st
import pandas as pd
import os
from model import train_customer_segmentation, load_segmentation_model, MODELS_DIR
from prediction import predict_customer_by_id, predict_from_features, predict_from_dataframe
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation - KMeans", layout="wide")
st.title("Customer Segmentation (KMeans) — Walmart")
st.markdown("Upload dataset transaksi Walmart (CSV) lalu klik *Train Segmentation*. Setelah itu lihat cluster, profil, dan prediksi cluster untuk customer baru.")

DATA_UP_PATH = "Walmart.csv"
MODEL_PATH = os.path.join(MODELS_DIR, "model_customer_kmeans.pkl")

# Sidebar: upload dataset for training
st.sidebar.header("Data & Training")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (transaction-level) untuk train segmentation", type=["csv"])
if uploaded:
    with open(DATA_UP_PATH, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Dataset disimpan ke {DATA_UP_PATH}")

n_clusters = st.sidebar.slider("Jumlah cluster (K)", min_value=2, max_value=10, value=4)

if st.sidebar.button("Train Segmentation"):
    try:
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

# Main area: show model results if available
if os.path.exists(MODEL_PATH):
    scaler, kmeans, customer_df, feature_cols = load_segmentation_model(MODEL_PATH)
    st.header("Cluster Overview")
    st.write(f"Found {customer_df.shape[0]} customers; features used: {feature_cols}")

    # basic cluster counts
    st.subheader("Cluster counts")
    counts = customer_df["cluster"].value_counts().sort_index()
    st.bar_chart(counts)

    # show cluster centers via PCA 2D
    st.subheader("2D visualization (PCA) of customers by cluster")
    try:
        pca = PCA(n_components=2)
        X = scaler.transform(customer_df[feature_cols].fillna(0).values)
        X2 = pca.fit_transform(X)
        vis_df = pd.DataFrame(X2, columns=["pc1", "pc2"])
        vis_df["cluster"] = customer_df["cluster"].values
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=vis_df, x="pc1", y="pc2", hue="cluster", palette="tab10", ax=ax)
        ax.set_title("PCA projection colored by cluster")
        st.pyplot(fig)
    except Exception as e:
        st.write("Visualisasi PCA gagal:", e)

    # show sample customers per cluster
    st.subheader("Sample customer profiles per cluster")
    cluster_sel = st.selectbox("Pilih cluster untuk melihat sample", sorted(customer_df["cluster"].unique()))
    sample_df = customer_df[customer_df["cluster"]==cluster_sel].head(10)
    st.dataframe(sample_df)

    # download model button
    with open(MODEL_PATH, "rb") as f:
        st.download_button("Download segmentation model (.pkl)", data=f, file_name="model_customer_kmeans.pkl")

    st.markdown("---")
    st.header("Prediksi cluster untuk customer baru")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediksi dari customer_id (jika sudah ada di profil)")
        cust_id_input = st.text_input("Masukkan customer_id (eks: 12345)")
        if st.button("Cari customer_id"):
            if cust_id_input:
                try:
                    prof = predict_customer_by_id(cust_id_input)
                    if prof is None:
                        st.info("Customer_id tidak ditemukan di profil. Kamu bisa prediksi dari fitur di kanan.")
                    else:
                        st.write("Customer profile dari training data:")
                        st.json(prof)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Masukkan customer_id.")

    with col2:
        st.subheader("Prediksi dari fitur (input manual atau CSV)")
        st.markdown("Masukkan JSON/dict yang berisi fitur customer sesuai kolom: " + ", ".join(feature_cols))
        json_input = st.text_area("Masukkan JSON/dict (contoh): {\"total_qty\": 100, \"avg_price\": 12.5, \"n_transactions\": 5}", height=120)
        if st.button("Predict from JSON"):
            if not json_input:
                st.warning("Masukkan JSON/dict di atas")
            else:
                try:
                    import json
                    d = json.loads(json_input)
                    res = predict_from_features(d)
                    st.success(f"Predicted cluster: {res['cluster']}")
                    st.write("Distances to cluster centers:", res["distances"])
                except Exception as e:
                    st.error(f"Error parsing/predicting: {e}")

        st.markdown("atau upload CSV berisi customer-level features")
        csv_up = st.file_uploader("Upload customer CSV (features)", type=["csv"], key="pred_csv")
        if csv_up is not None:
            try:
                dfc = pd.read_csv(csv_up)
                out = predict_from_dataframe(dfc)
                st.dataframe(out.head(20))
                # allow download
                st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="customer_clusters.csv")
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Model segmentation belum tersedia. Upload dataset di sidebar lalu klik 'Train Segmentation'.")

st.markdown("---")
st.write("Notes:")
st.write("""
- Pastikan dataset mengandung kolom 'customer_id' dan setidaknya 'quantity_sold' atau 'unit_price' agar fitur agregat bermakna.
- Feature names used in prediction: """ + (", ".join(feature_cols) if os.path.exists(MODEL_PATH) else "(n/a)") )

