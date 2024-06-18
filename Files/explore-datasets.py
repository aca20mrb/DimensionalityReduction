from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd

def print_samples(dataset, name, features, target):
    # Create a DataFrame from the dataset
    data = pd.DataFrame(dataset.data, columns=features)
    data['Target'] = dataset.target
    # Print the first 5 samples
    print(f"First 5 samples from the {name} dataset:")
    print(data.head(5))
    print("\n")

# Load Iris dataset
iris = load_iris()
print_samples(iris, 'Iris', iris.feature_names, 'Species')

# Load California Housing dataset
california_housing = fetch_california_housing()
print_samples(california_housing, 'California Housing', california_housing.feature_names, 'Median House Value')

