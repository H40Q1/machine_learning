# EX 8 Anomaly detection

import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# 1.1 Data preprocess---------------------------------------------------------------------------
mat = sio.loadmat('ex8data1.mat')  # contains 'X', 'Xval', 'yval'
X = mat.get('X')  # (307,2)
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
                                            mat.get('yval').ravel(),  # ravel y
                                            test_size=0.5)  # use train_test_split to split val into train and test


# 1.2 Fit parameters-------------------------------------------------------------
mu = X.mean(axis=0)  # (2,) mu of each feature
cov = np.cov(X.T)  # (2,2)  cov of each feature


# 1.3 Probability model-------------------------------------------------------------
multi_normal = stats.multivariate_normal(mu, cov)  # create multi-var Gaussian model
x, y = np.mgrid[0:30:0.01, 0:30:0.01]  # this mgrid creates two row and col both from 0 to 30 matrices X=y= (3000,3000)
pos = np.dstack((x, y))  # (3000,3000,2)  dstack stacks two array


# plot probability density
fig, ax = plt.subplots()
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')  # use (3000,3000,2) matrix x and y as two axis to plot contour

# plot original data points
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
           fit_reg=False,
           ax=ax,
           scatter_kws={"s":10,
                        "alpha":0.4})
# plt.show()


# 1.4 Select threshold epsilon-------------------------------------------------------------
# use cv set to select best epsilon

def select_threshold(X, Xval, yval):
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)  # create model with mu and cov from train data

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(Xval)  # input cv data X to predict

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')  # determin whether anomaly (>e) or not (<= e)
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]  # return best epsilon with best f1 score


e, fs = select_threshold(X, Xval, yval)  # best epsilon



# 1.5 Predict on test set------------------------------------------------------------------
def predict(X, Xval, e, Xtest, ytest):
    # use tain and CV data to compute parameters to build model
    # use CV data to fit best threshold
    Xdata = np.concatenate((X, Xval), axis=0)  # combine train and cv data

    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)  # model

    # calculate probability of test data
    pval = multi_normal.pdf(Xtest)  # (m,1)
    y_pred = (pval <= e).astype('int')

    print(classification_report(ytest, y_pred))

    return multi_normal, y_pred

multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)




