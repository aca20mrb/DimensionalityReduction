import pandas as pd
import ast

# Load the CSV file
df = pd.read_csv('new_optical_labels.csv')

# Function to convert target field to a single integer
def convert_target(target_str):
    target_list = ast.literal_eval(target_str)  # Convert the string representation of the list to an actual list
    return target_list.index(1)  # Find the index of the element that is 1

# Apply the conversion function to the target column
df['target'] = df['target'].apply(convert_target)

# Save the modified DataFrame to a new CSV file
df.to_csv('optical_labels.csv', index=False)

print("Conversion completed and saved to 'optical_labels.csv'.")
