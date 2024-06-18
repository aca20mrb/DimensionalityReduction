import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X = housing.data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define ranges for parameters
component_range = [1, 2, 3]
k_range = [2, 3, 4, 5, 6]

# Prepare data containers
k_results = []
component_results = []

# Iterate over different numbers of components for dimensionality reduction methods
for n_components in component_range:
    methods = {
        'NR': X_scaled,
        'PCA': PCA(n_components=n_components).fit_transform(X_scaled),
        'K-PCA': KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X_scaled),
        'RP': GaussianRandomProjection(n_components=n_components).fit_transform(X_scaled),
        't-SNE': TSNE(n_components=min(n_components, 3)).fit_transform(X_scaled)
    }

    # Clustering and silhouette scores for varying K
    for n_clusters in k_range:
        for name, X_reduced in methods.items():
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, clusters)
            k_results.append({'Method': name, 'K-Value': n_clusters, 'Silhouette Score': score})

    # Evaluate the impact of component change for a fixed K-value (e.g., K=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    for name, X_reduced in methods.items():
        clusters = kmeans.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, clusters)
        component_results.append({'Method': name, 'Components': n_components, 'Silhouette Score': score})

# Set seaborn style to match the example
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))  # Consistent figure size

# Plotting results for varying K
plt.subplot(1, 2, 1)
k_df = pd.DataFrame(k_results)
k_df.to_csv('housing_k_variation.csv', index=False)

sns.lineplot(data=k_df, x='K-Value', y='Silhouette Score', hue='Method', palette='Blues', ci=None)
plt.title('Silhouette Score vs. K-Value')
plt.xlabel('K-Value')
plt.ylabel('Silhouette Score')

# Plotting results for varying number of components
plt.subplot(1, 2, 2)
component_df = pd.DataFrame(component_results)
sns.lineplot(data=component_df, x='Components', y='Silhouette Score', hue='Method', palette='Blues', ci=None)
plt.title('Silhouette Score vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Silhouette Score')

plt.tight_layout()  # Adjust layout

plt.savefig('housing_comparisons_plot.png')
plt.show()
