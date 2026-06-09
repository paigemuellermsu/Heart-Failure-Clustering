import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from pathlib import Path

script_dir = Path(__file__).parent
csv_path = script_dir / "heart_failure_clinical_records_dataset_real.csv"

df = pd.read_csv(csv_path)

df.head()

X = df.drop(columns=["id", "DEATH_EVENT"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hc = AgglomerativeClustering(n_clusters=3)
df["Cluster_HC"] = hc.fit_predict(X_scaled)
df.head()

# Dendrogram
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10,5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# PCA Plot
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster_HC"],
    cmap="viridis"
)
plt.title("Hierarchical Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# Isomap Plot
from sklearn.manifold import Isomap

iso = Isomap(n_components=2)
X_iso = iso.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(
    X_iso[:,0],
    X_iso[:,1],
    c=df["Cluster_HC"],
    cmap="viridis"
)
plt.title("Hierarchical Clustering (Isomap Projection)")
plt.colorbar()
plt.show()

# t-SNE Plot
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=df["Cluster_HC"],
    cmap="viridis"
)
plt.title("Hierarchical Clustering (t-SNE Projection)")
plt.colorbar()
plt.show()

# Cluster Summary
cluster_summary = df.groupby("Cluster_HC").agg(
    n=("Cluster_HC", "size"),
    mean_age=("age", "mean"),
    sd_age=("age", "std"),
    pct_anaemia=("anaemia", lambda x: x.mean()*100)
)

print(cluster_summary)

# age --> age of patient
# anemia --> 0: false, 1: true
# creatinine_phosphokinase --> level of the enzyme in the blood
# diabetes --> 0: false, 1: true
# ejection_fraction --> percentage of blood leaving the heart at each contraction
# high_blood_pressure --> 0: false, 1: true
# platelets --> number of platelets in the blood
# serum_creatinine --> level of creatinine in the blood
# serum_sodium --> level of sodium in the blood
# sex --> 0: female, 1: male
# smoking --> 0: false, 1: true
# time --> follow-up time in days
# DEATH_EVENT --> 0: false, 1: true