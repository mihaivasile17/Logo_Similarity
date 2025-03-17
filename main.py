import numpy as np

# Load cluster results
clusters = np.load("logo_clusters.npy", allow_pickle=True).item()

# Sort clusters by cluster ID
sorted_clusters = sorted(clusters.items(), key=lambda x: x[0])

# Print results
print("Grouped websites by logo similarity:")
for cluster_id, websites in sorted_clusters:
    print(f"Cluster {cluster_id} ({len(websites)} logos): {websites}")
