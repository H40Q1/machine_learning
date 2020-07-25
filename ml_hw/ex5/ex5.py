# ex5 bias vs. variance

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1.1 Data preprocess--------------------------------------------------------------------------------
# 1.1.1 load data
def load_data():
    """for ex5
    d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])  # ravel for panda to plot


X, y, Xval, yval, Xtest, ytest = load_data()
# X, y train set (12,)
# Xval, yval cross validation set (21,)
# Xtest, ytest test set (21,)


# 1.1.2 plot data
# df = pd.DataFrame({'water_level': X, 'flow': y})
# sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
# plt.show()

# 1.1.3 add intercept x0=1
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
# X (12,2)
# Xval, Xtest (21,2)


# 1.1.4 init theta
theta = np.ones(X.shape[1])  # (n+1,)  n=1 j=2


# 1.2 Cost and grad-------------------------------------------------------------------------------------
# 1.2.1 linear regression cost
def cost(theta, X, y):  # theta(n,1) X(m,n) y(m,1)
    m = X.shape[0]
    inner = X @ theta - y  # (m*1)
    square_sum = sum(np.power(inner, 2))
    J = square_sum / (2 * m)
    return J


# 1.2.2 regularized cost
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:],
                                                2).sum()  # theta is an array (2,), calculate from j1 so theta[1:]
    return cost(theta, X, y) + regularized_term


# 1.2.3 gradient
# no update iteration
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # (m,n+1).T @ (m, 1) -> (n+1, 1)

    return inner / m  # return the gradient part, which is g* in (theta=theta-g*)


# 1.2.4 regularized gradient
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta (n+1,)
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term  # (n+1,1)


# 1.3 Fitting data-----------------------------------------------------------
# 1.3.1 linear regression
def linear_regression_np(X, y, l=1):
    # init theta
    theta = np.ones(X.shape[1])  # (n+1,)

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    # for opt minimize
    # no constraints on args
    return res


final_theta = linear_regression_np(X, y, l=0).get('x')

# 1.3.2 plot
b = final_theta[0]  # intercept
m = final_theta[1]  # slope

# plt.scatter(X[:, 1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1] * m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()


# 1.4 Analyze learning curves-----------------------------------------------------------
# 1.4.1 compute train cost and cv cost
training_cost, cv_cost = [], []
m = X.shape[0]
for i in range(1, m + 1):  # 1 to m
    #     print('i={}'.format(i))
    # X[:1, :] means only read X index 0 row
    res = linear_regression_np(X[:i, :], y[:i], l=1)  # put part of X and y until current i to train, increase m
    # for following, no regularization, l = 0
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)  # train cost with current optimal theta, no regularization
    cv = regularized_cost(res.x, Xval, yval, l=0)  # cv cost with current optimal theta, no regularization
    #     print('tc={}, cv={}'.format(tc, cv))

    training_cost.append(tc)
    cv_cost.append(cv)


# 1.4.2 plot
# plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
# plt.legend(loc=1)
# plt.show()

# conclusion: cost is large, its a high bias curve, caused by under fitting


# 1.5 Improvement ----------------------------------------------------------------------
# 1.5.1 functions to create more features
def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """

    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)  # every example in args extend from degree 1 to power
        # normalization
        ndarr = normalize_feature(df).values  # take out the datafram's value as a matrix
        # add intercept term after normalization
        return np.insert(ndarr, 0, ndarr.shape[0], axis=1)  # add intercept x0 = 1

    return [prepare(x) for x in args]  # return X (m,power+1)


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df  # return df


def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


X, y, Xval, yval, Xtest, ytest = load_data()  # load again
X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)  # increase to 8 degree, X_poly(m,9)


# 1.5.2 plot learning curves for current datasets
# no regularization part when plot lc, so l = 0

def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)
        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)


# plot_learning_curve(X_poly, y, Xval_poly, yval, l=0) # over-fitting, train cost is 0
# plt.show()

# 1.5.3 find optimal lambda
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)


plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()


print(l_candidate[np.argmin(cv_cost)])
