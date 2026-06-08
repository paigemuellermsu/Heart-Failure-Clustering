import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df.head()

X = df.drop(columns=["DEATH_EVENT"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hc = AgglomerativeClustering(n_clusters=3)
df["Cluster_HC"] = hc.fit_predict(X_scaled)
df.head()

Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10,5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()