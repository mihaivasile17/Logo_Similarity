from sklearn.cluster import KMeans
import numpy as np

# Load embeddings
features = np.load("logo_embeddings.npy", allow_pickle=True).item()

# Convert embeddings to a matrix
img_names = list(features.keys())
embeddings = np.array(list(features.values()))

# Apply K-Means clustering
num_clusters = 500
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# Organize results into clusters
clusters = {}
for img_idx, cluster_id in enumerate(labels):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(img_names[img_idx])

# Save cluster results
np.save("logo_clusters.npy", clusters)
print("K-Means clustering completed!")
