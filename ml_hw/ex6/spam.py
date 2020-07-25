# ex6 spam filter

from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy.io as sio


# 4.1 Data preprocess----------------------------------------------------------------
mat_tr = sio.loadmat('spamTrain.mat')
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()  # after ravel (4000,1899) (4000,)
mat_test = sio.loadmat('spamTest.mat')
test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()  # (1000,1899) (1000,)

# 4.2 SVM----------------------------------------------------------------------------
svc = svm.SVC()
svc.fit(X, y)  # fit the svc model
pred = svc.predict(test_X)  # use this model and x test to predict y test
print(metrics.classification_report(test_y, pred))


# 4.3 Logistic regression--------------------------------------------------------------
logit = LogisticRegression()
logit.fit(X, y)
pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))



