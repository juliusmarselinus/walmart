import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
...
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

st.markdown("---")
st.write("Notes:")
st.write("""
- Pastikan dataset mengandung kolom 'customer_id' ...
- Feature names used in prediction: """ + (", ".join(feature_cols) if os.path.exists(MODEL_PATH) else "(n/a)") )

