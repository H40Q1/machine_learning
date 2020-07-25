# ex1 linear regression with single feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.1 Dealing with data----------------------------------------------------------
# 1.1.1 read file
path = "ex1data1.txt"

# 2.1.2 write as a table
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()  # print(data.head())
data.describe()  # print(data.describe())

# 1.1.3 plot data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))  # plt.show()

# 1.1.4 Preprocess data
data = (data - data.mean()) / data.std()  # mean normalization
data.insert(0, 'Ones', 1)  # add x0=1 column, 0=column no, 1=val added, 'Ones' is header
cols = data.shape[1]  # get number of columns, which is n+2
X = data.iloc[:, 0:cols - 1]  # X is the all the columns except the last one
y = data.iloc[:, cols - 1:cols]  # Y is the last column
X = np.matrix(X.values)  # convert X and Y from csv table to numpy matrix, every xi given is (xi)T
y = np.matrix(y.values)
theta = np.zeros((cols-1, 1))  # init theta as 0, define it as (n+1)*1

# 1.2 Gradient Descent----------------------------------------------------------
# 1.2.1 compute cost function
def computeCost(X, y, theta):
    h = X @ theta  # h_theta is a 97*1 matrix of prediction  NOTE: avoid using *
    sq_error = np.power((h - y), 2)  # 97*1 matrix of square error
    result = sum(sq_error) / (2 * len(X))  # cost of given X, y, theta
    return result


# 1.2.2 batch gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)  # temp theta initiation
    cost = np.zeros(iters)  # cost init, 1*iters array
    j_lim = int(theta.ravel().shape[0])  # j_lim = n+1 , range(j_lim) = 0,1,2,..,j_lim-1

    for i in range(iters):
        h = X @ theta  # calculate prediction h for each example i
        error = h - y  # calculate error for each example i
        for j in range(j_lim):
            term = np.multiply(error, X[:, j])  # element-wise product, produce a 97*1 vector of term of current j
            temp[j, 0] = theta[j, 0] - ((alpha / len(X)) * np.sum(term))

        theta = temp  # update theta for each iteration
        cost[i] = computeCost(X, y, theta)  # record cost for each iteration with current theta

    return theta, cost

# init learning rate and no. of iterations
alpha = 0.01
iters = 1000

theta, cost = gradientDescent(X, y, theta, alpha, iters)
final_cost = cost[-1]

# 1.3 Visualization----------------------------------------------------------
# 1.3.1 Linear Fitting
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # x value
f = theta[0, 0] + (theta[1, 0] * x)  # prediceted value corresponds to x value
fig, ax = plt.subplots(figsize=(12, 8))  # create a subplot ax, because 1st plot is in 2.1.3
ax.plot(x, f, 'r', label='Prediction')  # plot linear line with red line
ax.scatter(data.Population, data.Profit, label='Traning Data')  # plot examples
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# 1.3.2 Cost function plot
fig, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(np.arange(iters), cost, 'r')  # plot x in range of iterations corresponds to cost[i]
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epoch')
plt.show()