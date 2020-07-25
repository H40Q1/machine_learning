# ex7 PCA

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

# 4.1 Data preprocess------------------------------------------------------------------------
# 4.1.1 load data
mat = sio.loadmat('ex7data1.mat')
X = mat.get('X')  # (50,2)


# 4.1.2 plot
# sns.lmplot('X1', 'X2',
#            data=pd.DataFrame(X, columns=['X1', 'X2']),
#            fit_reg=False)
# plt.show()


# 4.1.3 normalize data
def normalize(X):
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy


# 4.2 PCA-------------------------------------------------------------------------------------
# 4.2.1 covariance matrix
def covariance_matrix(X):
    m = X.shape[0]
    return (X.T @ X) / m


# 4.2.2 pca main
def pca(X):
    # 1. normalize data
    # X_norm = normalize(X)

    # 2. calculate covariance matrix
    Sigma = covariance_matrix(X)  # (n, n)

    # 3. do singular value decomposition
    # remeber, we feed cov matrix in SVD, since the cov matrix is symmetry so U = V
    U, S, V = np.linalg.svd(Sigma)  # U: principle components (n, n)

    return U, S, V


# 4.2.3 project data
def project_data(X, U, k):
    m, n = X.shape
    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]  # (m,n) * (n*k) = (m,k) return as new X



# 4.3 Test---------------------------------------------------------------------------
X_norm = normalize(X)
U, S, V = pca(X_norm)
Z = project_data(X_norm, U, 1)

# plot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot('X1', 'X2',
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax1)
ax1.set_title('Original dimension')

sns.rugplot(Z, ax=ax2)  # rugplot doesn't need y
ax2.set_xlabel('Z')
ax2.set_title('Z dimension')
plt.show()