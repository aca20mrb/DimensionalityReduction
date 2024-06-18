import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize((64, 64))  # Resize to a consistent size
                img_array = np.array(img)
                flattened_img = img_array.flatten()  # Flatten the image
                images.append(flattened_img)
    return np.array(images)  # Return as a 2D array

# Specify the folder containing the preprocessed images
image_folder = 'images/preprocessed_images'

# Load and standardize the data
data = load_images(image_folder)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply k-means clustering
n_clusters = 8  # Specify the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Plot the cluster centers as images
fig, axes = plt.subplots(1, n_clusters, figsize=(20, 5))
for i, ax in enumerate(axes):
    cluster_center = kmeans.cluster_centers_[i].reshape(64, 64)
    ax.imshow(cluster_center, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Cluster {i}')
plt.suptitle('Cluster Centers of Image Dataset')
plt.show()