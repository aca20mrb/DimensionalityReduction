import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X = housing.data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set parameters for dimensionality reduction
n_components = 1  # Number of dimensions to reduce to
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
n_clusters = 4  # Specified number of clusters, adjust based on your data analysis needs

# Prepare to collect scores for each method
results = {method: [] for method in methods}

# Perform multiple iterations
for _ in range(n_iterations):
    for name, method in methods.items():
        if name == 'NR':
            X_reduced = method  # Use the scaled data directly
        else:
            X_reduced = method.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, clusters)
        results[name].append(score)

# Prepare data for seaborn plotting
data = pd.DataFrame.from_dict(results).melt(var_name='Method', value_name='Silhouette Score')
print(data)
# Use seaborn to plot
sns.set(style="whitegrid", context='talk')
plt.figure(figsize=(14, 10))  # Adjust figure size to give more room
palette = sns.color_palette("Blues", n_colors=len(methods))
ax = sns.barplot(x='Method', y='Silhouette Score', data=data, capsize=.1, palette=palette, ci='sd')
ax.set_title('Comparison of Dimensionality Reduction Techniques with K-Means Clustering - California Housing', color='black')
ax.set_ylabel('Average Silhouette Score', color='black')
ax.set_xlabel('Methods', color='black')
ax.set_ylim(0,0.8)
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
# Add annotation for the number of components and K-value
textstr = f'Number of Components: {n_components}\nK-Value: {n_clusters}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# Place text on the upper right, adjust the position as necessary
ax.text(0.99, 0.99, textstr, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', horizontalalignment='right', bbox=props, color='black')
# Adjust layout to make room for label
plt.tight_layout(pad=2)  # Adjust the padding to ensure nothing gets cut off
# Show the plot
plt.savefig('housing_reduction_plot.png')
plt.show()