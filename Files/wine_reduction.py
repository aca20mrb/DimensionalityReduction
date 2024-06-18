import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Load data
wine = load_wine()
X = wine.data

print(X[0])
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set parameters for dimensionality reduction
n_components = 2  # Number of dimensions to reduce to
n_iterations = 1  # Number of iterations to average results

# Dimensionality reduction methods
methods = {
    'NR': X_scaled,  # Using the original scaled data
    'PCA': PCA(n_components=n_components),
    'K-PCA': KernelPCA(n_components=n_components, kernel='rbf', random_state=42),
    'RP': GaussianRandomProjection(n_components=n_components),
    't-SNE': TSNE(n_components=n_components, random_state=42)  # Added t-SNE
}

# K-means parameters
n_clusters = 3  # Specified number of clusters, adjust based on your data analysis needs

# Prepare to collect scores and processing times for each method
results = {method: {'Silhouette': [], 'Davies-Bouldin': [], 'Time': []} for method in methods}
cluster_maps = {}
# Perform multiple iterations
for _ in range(n_iterations):
    for name, method in methods.items():
        start_time = time.time()
        if name == 'NR':
            X_reduced = method  # Use the scaled data directly
        else:
            X_reduced = method.fit_transform(X_scaled)
        end_time = time.time()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_reduced)
        silhouette_avg = silhouette_score(X_reduced, clusters)
        davies_bouldin_avg = davies_bouldin_score(X_reduced, clusters)
        results[name]['Silhouette'].append(silhouette_avg)
        results[name]['Davies-Bouldin'].append(davies_bouldin_avg)
        results[name]['Time'].append((end_time - start_time)*1000)
        if name not in cluster_maps:
            cluster_maps[name] = (X_reduced, clusters)

# Print the Davies-Bouldin scores for each method
for method, scores in results.items():
    print(f"{method}:")
    print(f"  Silhouette Score: {np.mean(scores['Silhouette'])}")
    print(f"  Davies-Bouldin Score: {np.mean(scores['Davies-Bouldin'])}")
    print(f"  Processing Time: {np.mean(scores['Time'])} milliseconds")

# Prepare data for seaborn plotting
silhouette_data = pd.DataFrame({method: scores['Silhouette'] for method, scores in results.items()})
davies_bouldin_data = pd.DataFrame({method: scores['Davies-Bouldin'] for method, scores in results.items()})
time_data = pd.DataFrame({method: scores['Time'] for method, scores in results.items()})

silhouette_data = silhouette_data.melt(var_name='Method', value_name='Silhouette Score')
davies_bouldin_data = davies_bouldin_data.melt(var_name='Method', value_name='Davies-Bouldin Score')
time_data = time_data.melt(var_name='Method', value_name='Processing Time')

# Plot Silhouette Scores
sns.set(style="whitegrid", context='talk')
plt.figure(figsize=(14, 10))  # Adjust figure size to give more room
palette = sns.color_palette("Blues", n_colors=len(methods))
ax1 = sns.barplot(x='Method', y='Silhouette Score', data=silhouette_data, capsize=.1, palette=palette, ci='sd')
ax1.set_title('Comparison of Dimensionality Reduction Techniques with K-Means Clustering - Silhouette Score',
              color='black')
ax1.set_ylabel('Average Silhouette Score', color='black')
ax1.set_xlabel('Methods', color='black')
ax1.set_ylim(0, 0.8)
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
plt.tight_layout(pad=2)  # Adjust the padding to ensure nothing gets cut off
plt.savefig('iris_silhouette_plot.png')
plt.show()

# Plot Davies-Bouldin Scores
plt.figure(figsize=(14, 10))  # Adjust figure size to give more room
ax2 = sns.barplot(x='Method', y='Davies-Bouldin Score', data=davies_bouldin_data, capsize=.1, palette=palette, ci='sd')
ax2.set_title('Comparison of Dimensionality Reduction Techniques with K-Means Clustering - Davies-Bouldin Score',
              color='black')
ax2.set_ylabel('Average Davies-Bouldin Score', color='black')
ax2.set_xlabel('Methods', color='black')
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
plt.tight_layout(pad=2)  # Adjust the padding to ensure nothing gets cut off
plt.savefig('iris_davies_bouldin_plot.png')
plt.show()

# Plot Processing Time
plt.figure(figsize=(14, 10))  # Adjust figure size to give more room
ax3 = sns.barplot(x='Method', y='Processing Time', data=time_data, capsize=.1, palette=palette, ci='sd')
ax3.set_title('Comparison of Dimensionality Reduction Techniques - Processing Time', color='black')
ax3.set_ylabel('Processing Time (seconds)', color='black')
ax3.set_xlabel('Methods', color='black')
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
plt.tight_layout(pad=2)  # Adjust the padding to ensure nothing gets cut off
plt.savefig('iris_processing_time_plot.png')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

for method, (X_reduced, clusters) in cluster_maps.items():
    plt.figure(figsize=(8, 6))

    if X_reduced.shape[1] == 1:
        sns.scatterplot(x=X_reduced[:, 0], y=[0] * len(X_reduced), hue=clusters, palette='viridis', legend='full')
        plt.xlabel('Component A')
        plt.ylabel('')
        plt.yticks([])  # Remove y-axis values
    else:
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=clusters, palette='viridis', legend='full')
        plt.xlabel('Component A')
        plt.ylabel('Component B')
        plt.yticks([])  # Remove y-axis values

    plt.title(f'Wine Cluster map for {method} with K-Means')
    plt.xticks([])  # Remove x-axis values
    plt.legend(title='Cluster')
    plt.savefig(f'wineplots/cluster_map_{method}.png')
    plt.show()

