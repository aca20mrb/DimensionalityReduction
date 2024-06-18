import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Path to the folder containing images
image_folder = 'images/preprocessed_images'

# Load images from the folder
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

# Convert list to numpy array
data = np.array(data)

# Example: Applying PCA for dimensionality reduction
pca = PCA(n_components=50)
data_reduced = pca.fit_transform(data)

# Example: Clustering the images
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data_reduced)

# Plotting a few images from each cluster (for visualization)
fig, axs = plt.subplots(5, 5, figsize=(10, 10))  # 5x5 subplots
for i in range(5):  # number of clusters
    imgs = np.where(clusters == i)[0][:5]  # Get first 5 images in each cluster
    for j, img in enumerate(imgs):
        axs[i, j].imshow(data[img].reshape(100, 100), cmap='gray')
        axs[i, j].axis('off')
plt.show()
