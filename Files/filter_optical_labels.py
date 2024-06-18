import os
import pandas as pd

# Define the paths
image_folder = 'images/preprocessed_images'
csv_file = 'optical_labels.csv'
filtered_csv_file = 'filtered_optical_labels.csv'

# Load the CSV file
labels_df = pd.read_csv(csv_file)

# Get the list of filenames in the preprocessed_images directory
image_filenames = set(os.listdir(image_folder))

# Filter the DataFrame to include only rows where the filename is in the image_filenames set
filtered_labels_df = labels_df[labels_df['filename'].isin(image_filenames)]

# Extract ID and side from the filename
filtered_labels_df['ID'] = filtered_labels_df['filename'].apply(lambda x: int(x.split('_')[0]))
filtered_labels_df['side'] = filtered_labels_df['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

# Order the DataFrame by ID and then by side ('left' before 'right')
filtered_labels_df = filtered_labels_df.sort_values(by=['ID', 'side'], ascending=[True, True])

# Save the filtered and ordered DataFrame to a new CSV file
filtered_labels_df.to_csv(filtered_csv_file, index=False)

print(f"Filtered and ordered CSV file saved as {filtered_csv_file}")
