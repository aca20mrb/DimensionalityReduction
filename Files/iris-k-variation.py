import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X = iris.data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define ranges for parameters
component_range = [2, 3]  # Adjusted to avoid 1 component which may not be meaningful for PCA/K-PCA
k_range = [2, 3, 4, 5, 6]

# Prepare data container
k_results = []

# Iterate over different numbers of components for dimensionality reduction methods
for n_components in component_range:
    methods = {
        'NR': X_scaled,
        'PCA': PCA(n_components=n_components).fit_transform(X_scaled),
        'K-PCA': KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X_scaled),
        'RP': GaussianRandomProjection(n_components=n_components).fit_transform(X_scaled),
        't-SNE': TSNE(n_components=min(n_components, 3)).fit_transform(X_scaled)  # Ensure t-SNE uses correct n_components
    }

    # Clustering and silhouette scores for varying K
    for n_clusters in k_range:
        for name, X_reduced in methods.items():
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, clusters)
            k_results.append({'Method': name, 'K-Value': n_clusters, 'Silhouette Score': score})

# Convert results to dataframe
k_df = pd.DataFrame(k_results)

# Print dataframe to ensure it contains the expected values
print("K-Value Results:")
k_df.to_csv('iris_k_variation.csv', index=False)

# Set seaborn style to match the example
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))  # Consistent figure size

# Plotting results for varying K
sns.lineplot(data=k_df, x='K-Value', y='Silhouette Score', hue='Method', palette='Blues', ci=None)
plt.ylim(0, 1)  # Set y-axis limits
plt.title('Silhouette Score vs. K-Value')
plt.xlabel('K-Value')
plt.ylabel('Silhouette Score')

plt.tight_layout()  # Adjust layout

#plt.savefig('iris_comparisons_plot.png')
plt.show()