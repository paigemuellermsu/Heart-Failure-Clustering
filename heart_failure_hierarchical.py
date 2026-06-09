import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
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

hc = AgglomerativeClustering(n_clusters=3)
df["Cluster_HC"] = hc.fit_predict(X_scaled)
df.head()

Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10,5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

cluster_summary = df.groupby("Cluster_HC").agg(
    n=("Cluster_HC", "size"),
    mean_age=("age", "mean"),
    sd_age=("age", "std"),
    pct_anaemia=("anaemia", lambda x: x.mean()*100)
)

print(cluster_summary)