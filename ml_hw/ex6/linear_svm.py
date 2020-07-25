# EX6 linear svm

import pandas as pd
import sklearn.svm
import scipy.io as sio
import matplotlib.pyplot as plt

# 1.1 Data preprocess--------------------------------------------------------
# 1.1.1 load data
mat = sio.loadmat('ex6data1.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

# 1.1.2 plot
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')  # s=area cmap=color
# ax.set_title('Raw data')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# plt.show()


# 1.2 SVM----------------------------------------------------------------------
# 1.2.1 try c=1
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')  # linear SVM
svc1.fit(data[['X1', 'X2']], data['y'])
svc1.score(data[['X1', 'X2']], data['y'])  # Return the mean accuracy on the given test data and labels

data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])  # Predict confidence scores for samples

# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=1) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

# try c=100

