# ex4 NN back propagation
# use a different kind of code from ex3 to realize NN model

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report  # this is the evaluation report lib

# 1.1 Data preprocess-----------------------------------------------------
# 1.1.1 load mat file
data = loadmat('ex4data1.mat')
X = data['X']  # (5000,400)
y = data['y']  # (5000,1)
X = np.matrix(X)
y = np.matrix(y)

# 1.1.2 One-Hot coding y (another way to one-hot coding y)
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)  # y_onehot (5000,10)


# [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.], index 0 is type 1...


# 1.2 Compute Cost via forward prop-----------------------------------------------------------------
# 1.2.1 sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 1.2.2 forward propagation
# X(5000,400) theta1(25,401) theta2 (10,26)
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # this time insert x0 at here   a1 (5000,401)
    z2 = a1 @ theta1.T  # (5000,401)*(401,25) = (5000,25)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 @ theta2.T  # (5000,26)*(26,10) = (5000,10)
    h = sigmoid(z3)  # h = a3 = (5000,10)

    return a1, z2, a2, z3, h


# 1.2.3 Cost function
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    # X = np.matrix(X)  # X(5000,401)
    # y = np.matrix(y)  # y(5000,10)   y here is y_onehot
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  # feed forward to get prediction h with init thetas

    # compute the cost
    J = 0
    for i in range(m):  # i = 0 ... m-1  python index
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))  # element wise multiply, y cols rank 0 to k
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)  # sum 1 to m

    J = J / m

    # add the cost regularization term
    # ignore theta 0, so start from col 1:
    # power both theta
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J


# 1.2.4 init data
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# random from 0 to 1 of theta1 and theta 2 then -0.5 then * 0.25 to make it close to 0
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
# len(params) = 10*26 + 401*25
# reshape the parameter array into parameter matrices for each layer
# take params's first (hidden_size * (input_size + 1)) as theta1 ,shape (hidden_size, (input_size + 1)) (25,401)
# take reset as theta 2 , shape (num_labels, (hidden_size + 1)) (10,26)

# theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
# theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))


# print(cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate, theta1, theta2))


# 1.3 Back propagation------------------------------------------------------------
# 1.3.1 sigmoid inverse
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 1.3.2 back prop
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # init
    m = X.shape[0]
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)


    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)  # compute cost
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  # get a1 z2 a2 z3 h with initiated theta

    for t in range(m):  # for each example extract a1 z2 a2 h
        a1t = a1[t, :]  # (1, 401) extract a1 with current t
        z2t = z2[t, :]  # (1, 25)  extract z2 with current t
        a2t = a2[t, :]  # (1, 26)  extract a2 with current t
        ht = h[t, :]  # (1, 10)    extract h with current t
        yt = y[t, :]  # (1, 10)    current example t's y_onehot

        d3t = ht - yt  # (1, 10)  layer 3 delta with current t

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26) insert z0 = 0
        # theta2 (10,26)    d3t(1,10)      sigmoid_gradient(z2t) (1,26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26) layer 2 delta
        # no d1t

        # delta l = delta l + layer l+1 delta of t * layer l a   !NOTE: a no a0
        delta1 = delta1 + (d2t[:, 1:]).T * a1t  # delta1 += layer 2 delta @ layer 1 a, ignore a0
        delta2 = delta2 + d3t.T * a2t  # delta2 += layer 3 delta @ layer 2 a, no a0 this time

    delta1 = delta1 / m  # calculate mean
    delta2 = delta2 / m

    # add the gradient regularization term, no regularization part on delta j=0
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # flat the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))  # (10285,)

    return J, grad


# J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)


# 1.4 find optimal----------------------------------------------------------------------------
# not learning  why?

fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels,
                                               X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
# it means if your backprop function return (cost, grad), you could set jac=True
# fmin.X is the optimal param

# reshape theta1 and 2
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print(cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))



# a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  # forward feed to find prediction h
# y_pred = np.array(np.argmax(h, axis=1) + 1)  # get the type of prediction h, change 0-index 0 to 1-index so +1
# print(y_pred.shape)
#
# correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
# accuracy = (sum(map(int, correct)) / float(len(correct)))
# print('accuracy = {0}%'.format(accuracy * 100))



# print(classification_report(y, y_pred))
