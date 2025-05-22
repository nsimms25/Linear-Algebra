import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
data = iris.data  # features only
feature_names = iris.feature_names

df = pd.DataFrame(data, columns=feature_names)
#print(df.head())

data_matrix = np.array(data)
#print(data_matrix.shape)

data_transposed = data_matrix.T

mat_mul = np.dot(data_matrix, data_transposed)

data_norm = data_matrix - np.mean(data_matrix, axis=0)

cov_matrix = np.cov(data_norm.T)

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eig_vals)
print("1st eigenvector:", eig_vecs[:, 0])
