# EX2 Logistic regression with no regularization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 1.1 Data Preprocess-------------------------------------------------------------
# 1.1.1 read file
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

# 1.1.2 plot data
positive = data[data['Admitted'].isin([1])]  # mark data which has "admitted" in 1 as positive dataset
negative = data[data['Admitted'].isin([0])]  # mark data which has "admitted" in 0 as negative dataset
fig, ax = plt.subplots(figsize=(12, 8))  # create plot
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  # plot positive
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  # plot negative
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
# plt.show()

# 1.1.3 split X and y
data.insert(0, 'Ones', 1)
cols = data.shape[1]  # cols = n+2
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
X = np.array(X.values)  # convert X and Y from csv table to numpy array for opt.fmin_tnc , every xi given is (xi)T
y = np.array(y.values)
theta = np.zeros(cols - 1)  # theta shape 1*(n+1)


# 1.2 Logistic regression-------------------------------------------------------------
# 1.2.1 Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # return sigmoid number


# 1.2.2 Cost function
def computeCost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X @ theta.T)))  # vector of m*1
    second = np.multiply((1 - y), np.log(1 - sigmoid(X @ theta.T)))  # vector of m*1
    return np.sum(first - second) / (len(X))


# 1.2.3 batch gradient descent (only take one iteration of gradient with init theta)
def gradientDescent(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    j_lim = int(theta.ravel().shape[1])  # j_lim = n+1 , range(j_lim) = 0,1,2,..,j_lim-1
    grad = np.zeros(j_lim)  # init array 1*(n+1)
    error = sigmoid(X @ theta.T) - y  # m*1 vector of errors

    for i in range(j_lim):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)  # grad is a n+1 array of n+1=j_lim sums with theta = zero matrix

    return grad
    #
    #
    # m, j = np.shape(X)  # j=n+1
    # theta = theta.reshape((j, 1))
    # grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y)/m
    # return grad.flatten()



# 1.2.4 use SciPy's truncated newton (TNC) to find optimal  (FAILED)
result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradientDescent, args=(X, y))
# print(result)




# # 1.2.5 normal batch gradient descent
# def gradientDescent(X, y, theta, alpha, iters):
#     j_lim = int(theta.ravel().shape[0])  # j_lim = n+1 , range(j_lim) = 0,1,2,..,j_lim-1
#     cost = np.zeros(iters)  # cost init, 1*iters array
#     temp = np.zeros(theta.shape)  # temp theta initiation (n+1)*1
#
#     for i in range(iters):
#         error = sigmoid(X @ theta) - y  # m*1 vector of errors
#
#         for j in range(j_lim):
#             term = np.multiply(error, X[:, j])
#             temp[j, 0] = theta[j, 0] - alpha * np.sum(term) / len(X)
#
#         theta = temp  # update theta for each iteration
#         cost[i] = computeCost(X, y, theta)  # record cost for each iteration with current theta
#
#     return theta, cost
#
#
# alpha = 0.01
# iters = 100000
#
# theta, cost = gradientDescent(X, y, theta, alpha, iters)
# final_cost = cost[-1]
#



# 1.3 Test-------------------------------------------------------------
# 1.3.1 Predict function
def predict(optimal_theta, X):
    probability = sigmoid(X @ optimal_theta.T)  # probability is the a m*1 prediction vector
    return [1 if x >= 0.5 else 0 for x in probability]


# 1.3.2 test accuracy
optimal_theta = np.matrix(result[0])  # result element 0 is the optimal theta
predictions = predict(optimal_theta, X)  # prediction according to optimal theta
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
