import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# Load data
image_folder = 'images/preprocessed_images'
labels_data = pd.read_csv('filtered_optical_labels.csv')
y = labels_data['target'].to_numpy()
def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize((128, 128))  # Resize
                img_array = np.array(img)
                flattened_img = img_array.flatten()  # Flatten the image
                images.append(flattened_img)
                filenames.append(filename)
    return np.array(images)


data = load_images(image_folder)
X = data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set parameters for dimensionality reduction
n_components = 5  # Number of dimensions to reduce to
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
n_clusters = 8  # Specified number of clusters


# Prepare to collect scores for each method
results = {method: {'Silhouette': [], 'Davies-Bouldin': [], 'NMI': [], 'Time': []} for method in methods}
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
        rand_avg = adjusted_rand_score(y, clusters)
        results[name]['Silhouette'].append(silhouette_avg)
        results[name]['Davies-Bouldin'].append(davies_bouldin_avg)
        results[name]['NMI'].append(rand_avg)
        results[name]['Time'].append((end_time - start_time) * 1000)
        if name not in cluster_maps:
            cluster_maps[name] = (X_reduced, clusters)
        print('Clusters: ', clusters)
        print('Y Target: ', y)
# Print the scores for each method
for method, scores in results.items():
    print(f"{method}:")
    print(f"  Silhouette Score: {np.mean(scores['Silhouette'])}")
    print(f"  Davies-Bouldin Score: {np.mean(scores['Davies-Bouldin'])}")
    print(f"  NMI: {np.mean(scores['NMI'])}")
    print(f"  Processing Time: {np.mean(scores['Time'])} milliseconds")
