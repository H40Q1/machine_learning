# EX 8 recommend system

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))
import numpy as np
import scipy.io as sio
import scipy.optimize as opt

# 2.1 Data preprocess --------------------------------------------------------------
# X (1682,10) movies with freathers
# theta (943,10) users' features
# Y (1682,943) num_movies * num_users matrix of user ratings of movies
# R (1682,943) num_movies * num_users matrix, where R(i, j) = 1 if the % i-th movie was rated by the j-th user

movies_mat = sio.loadmat('ex8_movies.mat')
Y, R = movies_mat.get('Y'), movies_mat.get('R')
param_mat = sio.loadmat('ex8_movieParams.mat')
theta, X = param_mat.get('Theta'), param_mat.get('X')


# 2.2 Cost + gradient---------------------------------------------------------------
# 2.2.1 serialize
def serialize(X, theta):
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))  # easy for computing regularized term


# 2.2.2 deserialize (reshape)
def deserialize(param, n_movie, n_user, n_features):
    # into ndarray of X(1682, 10), theta(943, 10)
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
           param[n_movie * n_features:].reshape(n_user, n_features)


# 2.2.3 Cost function
def cost(param, Y, R, n_features):
    # compute cost for every r(i, j)=1
    # param: serialized X, theta
    # Y (movie, user), (1682, 943): (movie, user) rating
    # R (movie, user), (1682, 943): (movie, user) has rating
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # element multiply R for all rated scores  (1682, 943)

    return np.power(inner, 2).sum() / 2


def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)
    return cost(param, Y, R, n_features) + reg_term


# 2.2.4 Gradient
def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)
    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param  # (1682* 10 + 943 *10)

    return grad + reg_term  # (1682* 10 + 943 *10))


# 2.3 Train---------------------------------------------------------------
ratings = np.zeros(1682)  # init ratings of new users = all 0
Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0 ,  Y (1682,944)
R = np.insert(R, 0, ratings != 0, axis=1)  # no rates for all the movies R (1682,944)

n_features = 50
n_movie, n_user = Y.shape
l = 10
X = np.random.standard_normal((n_movie, n_features))  # init new random X (1682, 50)
theta = np.random.standard_normal((n_user, n_features))  # init new random theta (944,50)
param = serialize(X, theta)

Y_norm = Y - Y.mean()

# normalized Y as args
res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=regularized_gradient)

X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)  # (1682, 50), (944, 50)


# prediction
prediction = X_trained @ theta_trained.T  # (1682,944)


# prediction of new users
my_preds = prediction + Y.mean()
