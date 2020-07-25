# EX2 Logistic regression with regularization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 2.1 Data Preprocess-------------------------------------------------------------
# 2.1.1 read file
path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()


# 2.1.2 plot data
positive = data[data['Admitted'].isin([1])]  # mark data which has "admitted" in 1 as positive dataset
negative = data[data['Admitted'].isin([0])]  # mark data which has "admitted" in 0 as negative dataset
fig, ax = plt.subplots(figsize=(12, 8))  # create plot
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  # plot positive
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  # plot negative
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
# plt.show()


# 2.1.3 Create high degree poly freatures
degree = 5  # polynomial degree
x1 = data['Exam 1']
x2 = data['Exam 2']
data.insert(3, 'Ones', 1)  # insert in columns 3, after x1 x2 y, later will become second column which is x0

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)  # create high degree poly features with x1x2

data.drop('Exam 1', axis=1, inplace=True)  # delete x1
data.drop('Exam 2', axis=1, inplace=True)  # delete x1


# 2.1.4 Split X, y, init theta
cols = data.shape[1]  # cols = n+2 = 12
X = data.iloc[:, 1:cols]   # columns except 0 are x
y = data.iloc[:, 0:1]  # column 0 is y
X = np.array(X.values)  # X y args has to be inputted as array
y = np.array(y.values)
theta = np.zeros(11)  # theta 1*(n+1)      h_theta = X @ theta.T
# X = np.matrix(X.values)  # convert X and Y from csv table to numpy matrix, every xi given is (xi)T
# y = np.matrix(y.values)
# theta = np.zeros((cols - 1, 1))  # theta shape (n+1)*1 = 11*1


# 2.2 Logistic regression with regularization-----------------------------------------------------------
# 2.2.1 Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # return sigmoid number

# 2.2.2 Cost function
def computeCost(theta, X, y, reg_lambda):
    theta = np.matrix(theta)  # matrix theta X and y in function
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X @ theta.T)))  # vector of m*1
    second = np.multiply((1 - y), np.log(1 - sigmoid(X @ theta.T)))  # vector of m*1
    third = reg_lambda * np.sum(np.power(theta, 2)) / (2*len(X))  # regularization part
    return np.sum(first - second) / (len(X)) + third

# 2.2.3 batch gradient descent (only take one iteration of gradient with init theta)
def gradientReg(theta, X, y, reg_lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    j_lim = int(theta.ravel().shape[1])
    grad = np.zeros(j_lim)
    error = sigmoid(X @ theta.T) - y  # m*1 vector of errors

    for ii in range(j_lim):
        term = np.multiply(error, X[:, ii])
        if ii == 0:
            grad[ii] = np.sum(term) / len(X)  # when j=0
        else:
            grad[ii] = (np.sum(term) / len(X)) + ((reg_lambda / len(X)) * theta[:, ii])  # when j != 0

    return grad


# 2.2.4 use SciPy's truncated newton (TNC) to find optimal
reg_lambda = 1
result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradientReg, args=(X, y, reg_lambda))
# x0 has to be an 1D array
# X y args has to be inputted as array
# theta has to be 1st args for both functions


# 1.3 Test-------------------------------------------------------------
# 1.3.1 Predict function
def predict(optimal_theta, X):
    probability = sigmoid(X @ optimal_theta.T)  # probability is the a m*1 prediction vector
    return [1 if x >= 0.5 else 0 for x in probability]  # prediction h_theta > 0.5 regard as 1


# 1.3.2 test accuracy
optimal_theta = np.matrix(result[0])  # result element 0 is the optimal theta
predictions = predict(optimal_theta, X)  # prediction according to optimal theta
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))


