# ex6 Gaussian kernels

import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio


# 2.1 Gaussian kernels ---------------------------------------------------------------
# kernel function
def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))


# 2.2 Data Preprocess----------------------------------------------------------------------------
# 2.2.1 load data
mat = sio.loadmat('ex6data2.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

# 2.2.2 plot
sns.set(context="notebook", style="white", palette=sns.diverging_palette(240, 10, n=2))
sns.lmplot('X1', 'X2', hue='y', data=data,
           size=5,
           fit_reg=False,
           scatter_kws={"s": 10}
          )
# plt.show()

# 2.3 SVM ----------------------------------------------------------------------------------------------
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)  # non-linear SVM
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])  # mean accuracy


predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 1]  # predict_proba return ndarray (data size, class)
# use [:, 1] or [:, 0] to define the type we want classify out
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')  # c means the type want to classify out
plt.show()