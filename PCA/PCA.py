import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()

X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

class PCA:
    
  def __init__(self,n_component):
    self.n_component = n_component

  def Stand_data(self,X):

    mu = np.sum(X,axis  =0)/X.shape[0]
    std = (np.sum((X-mu)**2,axis =0)/(X.shape[0]-1))**(0.5)
    return (X-mu)/std

  ###### Eigenvalues and vectors  of the covariance matrix for the stand data

  def eigs(self,X):
    X = self.Stand_data(X)
    Cov = 1/(X.shape[0]-1)*X.T@X
    eiV , eiVec = np.linalg.eig(Cov)
    # idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
    # print(idx)


    # eigen_values_sorted = eigen_values[idx]
    # eigen_vectors_sorted = eigen_vectors.T[:,idx]


    return np.linalg.eig(Cov)



  def Data_project(self,U,X):

    U =  U.T
    P = U[: self.n_component,: ]


    return X@P.T



  def fit(self,X):

    X = self.Stand_data(X)

    return  self.eigs(X)[1]

  def transform(self,X):

    return self.Data_project(self.fit(X),X)


my_pca = PCA(n_component=2)
my_pca.fit(X)
new_X = my_pca.transform(X)


print(f" the shape of the new data = {new_X.shape}")




plt.scatter(new_X[:,0], new_X[:,1],c=y)
plt.show()