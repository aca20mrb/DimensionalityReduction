import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the images and true labels
def load_images_and_labels(image_folder, csv_file, max_images):
    image_list = []
    labels = []
    filenames = []

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Filter to only include left eye images
    df_left_eye = df[df['side'] == 'left']

    for i, row in df_left_eye.iterrows():
        if i >= max_images:
            break
        filename = row['filename']
        label = row['target']

        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure image is in RGB format
            img = img.resize((128, 128))  # Ensure image is 512x512
            img_array = np.array(img).flatten()  # Flatten the image to a 1D array
            image_list.append(img_array)
            labels.append(label)
            filenames.append(filename)

    return np.array(image_list), np.array(labels), filenames


# Load the images and labels
image_folder = 'images/preprocessed_images'
csv_file = 'filtered_optical_labels.csv'
images, true_labels, filenames = load_images_and_labels(image_folder, csv_file, max_images=1000)

# Check if images are loaded correctly
if images.size == 0:
    raise ValueError("No images were loaded. Please check the image folder and CSV file.")

# Step 2: Normalize the data
scaler = StandardScaler()
images_normalized_scaled = scaler.fit_transform(images)

# Ensure the normalized images array is 2D
if len(images_normalized_scaled.shape) != 2:
    raise ValueError("The normalized images array is not 2D. Please check the image loading and preprocessing steps.")

# Define a function to perform clustering and calculate Rand Index
def cluster_and_evaluate(images_reduced, true_labels, num_clusters=5):
    centroids, _ = kmeans(images_reduced.astype(float), num_clusters)
    cluster_labels, _ = vq(images_reduced, centroids)
    rand_index = rand_score(true_labels, cluster_labels)
    return rand_index, cluster_labels

# Function to run the clustering and evaluation multiple times and return the average Rand Index
def run_experiment(images, true_labels, num_clusters=5, n_times=10):
    rand_indices_nr = []
    rand_indices_pca = []
    rand_indices_kpca = []
    rand_indices_tsne = []
    rand_indices_rp = []

    for _ in range(n_times):
        # NR (Non-Reduced)
        rand_index_nr, _ = cluster_and_evaluate(images, true_labels, num_clusters)
        rand_indices_nr.append(rand_index_nr)

        # PCA
        pca = PCA(n_components=n_components)
        images_pca = pca.fit_transform(images)
        rand_index_pca, _ = cluster_and_evaluate(images_pca, true_labels, num_clusters)
        rand_indices_pca.append(rand_index_pca)

        # Kernel PCA
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        images_kpca = kpca.fit_transform(images)
        rand_index_kpca, _ = cluster_and_evaluate(images_kpca, true_labels, num_clusters)
        rand_indices_kpca.append(rand_index_kpca)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)  # t-SNE typically uses 2 or 3 components
        images_tsne = tsne.fit_transform(images)
        rand_index_tsne, _ = cluster_and_evaluate(images_tsne, true_labels, num_clusters)
        rand_indices_tsne.append(rand_index_tsne)

        # Random Projection
        rp = SparseRandomProjection(n_components=n_components, random_state=42)
        images_rp = rp.fit_transform(images)
        rand_index_rp, _ = cluster_and_evaluate(images_rp, true_labels, num_clusters)
        rand_indices_rp.append(rand_index_rp)

    avg_rand_index_nr = np.mean(rand_indices_nr)
    avg_rand_index_pca = np.mean(rand_indices_pca)
    avg_rand_index_kpca = np.mean(rand_indices_kpca)
    avg_rand_index_tsne = np.mean(rand_indices_tsne)
    avg_rand_index_rp = np.mean(rand_indices_rp)

    return avg_rand_index_nr, avg_rand_index_pca, avg_rand_index_kpca, avg_rand_index_tsne, avg_rand_index_rp

# Step 3: Apply dimensionality reduction techniques and evaluate
num_clusters = 8

# Determine the number of components for PCA and other DR techniques
n_samples, n_features = images_normalized_scaled.shape
#n_components = min(n_samples, n_features)
n_components = 2

# Run the experiment
n_times = 10
avg_rand_index_nr, avg_rand_index_pca, avg_rand_index_kpca, avg_rand_index_tsne, avg_rand_index_rp = run_experiment(images_normalized_scaled, true_labels, num_clusters, n_times)

print(f"Average Rand Index over {n_times} runs:")
print(f"NR: {avg_rand_index_nr}")
print(f"PCA: {avg_rand_index_pca}")
print(f"Kernel PCA: {avg_rand_index_kpca}")
print(f"t-SNE: {avg_rand_index_tsne}")
print(f"Random Projection: {avg_rand_index_rp}")
