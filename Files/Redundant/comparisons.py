import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target  # True class labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a function to perform clustering and return silhouette score
def perform_clustering(X_transformed):
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_transformed)
    silhouette_avg = silhouette_score(X_transformed, cluster_labels)
    return cluster_labels, silhouette_avg

# Define a function to calculate percentage accuracy
def calculate_accuracy(y_true, y_pred):
    unique_labels = np.unique(y_true)
    accuracy = 0
    for label in unique_labels:
        cluster_indices = np.where(y_pred == label)[0]
        true_label = np.argmax(np.bincount(y_true[cluster_indices]))
        accuracy += np.sum(y_true[cluster_indices] == true_label)
    return accuracy / len(y_true)

# Perform clustering without any dimensionality reduction
cluster_labels_original, silhouette_scores_original = perform_clustering(X_scaled)
accuracy_original = calculate_accuracy(y_true, cluster_labels_original)
print("Accuracy without dimensionality reduction:", accuracy_original)

# Perform clustering with PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
cluster_labels_pca, silhouette_scores_pca = perform_clustering(X_pca)
accuracy_pca = calculate_accuracy(y_true, cluster_labels_pca)
print("Accuracy with PCA:", accuracy_pca)

# Perform clustering with Truncated SVD
tsvd = TruncatedSVD(n_components=2, random_state=42)
X_tsvd = tsvd.fit_transform(X_scaled)
cluster_labels_tsvd, silhouette_scores_tsvd = perform_clustering(X_tsvd)
accuracy_tsvd = calculate_accuracy(y_true, cluster_labels_tsvd)
print("Accuracy with Truncated SVD:", accuracy_tsvd)

# Perform clustering with Random Projection
rp = GaussianRandomProjection(n_components=2, random_state=42)
X_rp = rp.fit_transform(X_scaled)
cluster_labels_rp, silhouette_scores_rp = perform_clustering(X_rp)
accuracy_rp = calculate_accuracy(y_true, cluster_labels_rp)
print("Accuracy with Random Projection:", accuracy_rp)

# Plotting the results
methods = ['Original', 'PCA', 'Truncated SVD', 'Random Projection']
scores = [silhouette_scores_original, silhouette_scores_pca, silhouette_scores_tsvd, silhouette_scores_rp]
accuracies = [accuracy_original, accuracy_pca, accuracy_tsvd, accuracy_rp]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Methods')
ax1.set_ylabel('Silhouette Score', color=color)
ax1.bar(methods, scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(methods, accuracies, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Comparison of Dimensionality Reduction Techniques for Clustering (Iris Dataset)')
plt.show()
