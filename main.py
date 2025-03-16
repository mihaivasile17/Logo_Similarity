import numpy as np

# Load cluster results
clusters = np.load("logo_clusters.npy", allow_pickle=True).item()

# Print results
print("Grouped Websites by Logo Similarity (K-Means with 200 Clusters):")
for cluster_id, websites in clusters.items():
    print(f"Cluster {cluster_id}: {websites}")
