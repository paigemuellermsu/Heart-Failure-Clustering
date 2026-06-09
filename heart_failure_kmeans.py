import os
print("WORKING DIRECTORY:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from pathlib import Path
import pandas as pd

script_dir = Path(__file__).parent
csv_path = script_dir / "heart_failure_clinical_records_dataset_real.csv"

df = pd.read_csv(csv_path)

df.head()

X = df.drop(columns=["id", "DEATH_EVENT"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters
df.head()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-Means Clusters")
plt.show()

cluster_summary = df.groupby("Cluster_HC").agg(
    n=("Cluster_HC", "size"),
    mean_age=("age", "mean"),
    sd_age=("age", "std"),
    pct_anaemia=("anaemia", lambda x: x.mean()*100)
)

print(cluster_summary)