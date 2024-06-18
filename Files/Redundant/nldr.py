import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for Nonlinear Dimensionality Reduction
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X_scaled)

# Apply K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# Compute the silhouette score
silhouette = silhouette_score(X_reduced, clusters)

# Plotting the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.title(f't-SNE followed by K-means clustering\nSilhouette Score: {silhouette:.2f}')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(scatter)
plt.show()
