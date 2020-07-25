# feed forward prediction with given thetas

import numpy as np
import scipy.io as sio  # used for load mat file
from sklearn.metrics import classification_report  # this is the evaluation report lib


# 1.1 Data preprocess--------------------------------------------------------------------
# 1.1.1 Load data & split X, y
def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector for later one-hot coding (5000,)

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


X, y = load_data('ex3data1.mat', transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # add x0 = 1 intercept
# X(5000,401)  y(5000,)


# 1.1.2 sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 1.1.3 load theta
def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('ex3weights.mat')
# theta1 (25,401) theta2 (10,26)
# three layer
# input = 400+1  hidden = 25+1 output = 10


# 1.2 Feed forward -----------------------------------------------------------
a1 = X
z2 = a1 @ theta1.T  # (5000,401)*(401,25) = (5000,25)
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)  # add the a0=1  (5000,26)
a2 = sigmoid(z2)  # (5000,26)
z3 = a2 @ theta2.T  # (5000,26)*(26,10) = (5000,10)
a3 = sigmoid(z3)  # a3 is the output (5000,10)

y_pred = np.argmax(a3, axis=1) + 1  # aixs represent rows,find the largest values' row index for 5000 examples
# y_pred (5000,)
print(classification_report(y, y_pred))



