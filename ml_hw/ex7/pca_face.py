# ex7 apply pca on face data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


# load data
mat = sio.loadmat('ex7faces.mat')
X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])  # reshape each xi 32*32 into 1024
# X(5000,1024)

# PCA
sk_pca = PCA(n_components=100)  # create model k=100
Z = sk_pca.fit_transform(X)  # fit Z (5000,100)

# recover
X_recover = sk_pca.inverse_transform(Z)  # X_revocer (5000,1024)
