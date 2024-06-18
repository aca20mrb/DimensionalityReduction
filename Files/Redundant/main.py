# This is a sample Python script.

# Press ⌃F5 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data.keys()

dimensionality_reduction_method = ''

#def reduce(method):
#    if method == 'PCA':
#        dimensionality_reduction_method = 'PCA'
#    elif method == 'TSVD':
#        dimensionality_reduction_method = 'TSVD'
#    elif method == 'RP':
#        dimensionality_reduction_method = 'RP'

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# construct a dataframe using pandas
df1=pd.DataFrame(data['data'],columns=data['feature_names'])

# Scale data before applying PCA
scaling=StandardScaler()

# Use fit and transform method
scaling.fit(df1)
Scaled_data=scaling.transform(df1)

# Set the n_components=3
principal=TruncatedSVD(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')

# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)

plt.show()