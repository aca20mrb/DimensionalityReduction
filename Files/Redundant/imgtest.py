import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
image_folder = 'images/preprocessed_images'

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize((100, 100))  # Resize
                img_array = np.array(img)
                flattened_img = img_array.flatten()  # Flatten the image
                images.append(flattened_img)
    return np.array(images)  # This will be a 2D array

data = load_images(image_folder)
X = data




# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set parameters for dimensionality reduction
n_components = 4  # Number of dimensions to reduce to
n_iterations = 1  # Number of iterations to average results

# Dimensionality reduction methods
methods = {
    'NR': X_scaled,  # Using the original scaled data
    'PCA': PCA(n_components=n_components),
    'K-PCA': KernelPCA(n_components=n_components, kernel='rbf', random_state=42),
    'RP': GaussianRandomProjection(n_components=n_components),
    #'t-SNE': TSNE(n_components=n_components, random_state=42)
}

# K-means parameters
n_clusters = 10 # Specified number of clusters

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

# Set seaborn style with dark background
sns.set(style="darkgrid", context='talk')
plt.figure(figsize=(14, 10))  # Adjust figure size to give more room
palette = sns.color_palette("Blues", n_colors=len(methods))
ax = sns.barplot(x='Method', y='Silhouette Score', data=data, capsize=.1, palette=palette, ci='sd')
ax.set_title('Comparison of Dimensionality Reduction Techniques with K-Means Clustering', color='white')
ax.set_ylabel('Average Silhouette Score', color='white')
ax.set_xlabel('Methods', color='white')
ax.set_ylim(0,0.8)
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')

# Add annotation for the number of components and K-value
textstr = f'Number of Components: {n_components}\nK-Value: {n_clusters}'
props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
# Place text on the upper right, adjust the position as necessary
ax.text(0.99, 0.99, textstr, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', horizontalalignment='right', bbox=props, color='white')

ax.figure.set_facecolor('#303030')  # Set a dark figure background
ax.patch.set_alpha(0.5)  # Set axis background to slightly transparent

# Adjust layout to make room for label
plt.tight_layout(pad=2)  # Adjust the padding to ensure nothing gets cut off

# Show the plot
plt.savefig('transparent_background_plot.png', transparent=True)
plt.show()
