# ex6 find best parameters

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio


# 3.1 Data Preprocess -----------------------------------------------------------------------
mat = sio.loadmat('ex6data3.mat')

training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])  # training set
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])  # cv set
cv['y'] = mat.get('yval')


# 3.2 Find Parameter (gridSearch) -----------------------------------------------------------------------
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

parameters = {'C': candidate, 'gamma': candidate}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)  # grid search, find the optimal parameters
clf.fit(training[['X1', 'X2']], training['y'])  # fit data


print(clf.best_params_)  # return best combination
print(clf.best_score_)  # return best score

ypred = clf.predict(cv[['X1', 'X2']])  # ypred returned by SVC
print(metrics.classification_report(cv['y'], ypred))
