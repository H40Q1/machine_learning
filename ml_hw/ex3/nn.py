# ex3 neural network 1-d

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio  # used for load mat file
import matplotlib
import scipy.optimize as opt


# from sklearn.metrics import classification_report  # this is the evaluation report lib


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


X, y = load_data('ex3data1.mat')


# 1.1.2 Plot an image
def plot_an_image(image):
    # images = 1*400
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)  # 400 features array reshape to 20*20
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))


pick_one = np.random.randint(0, 5000)  # randomly pick one example


# plot_an_image(X[pick_one, :])
# plt.show()
# print('this should be {}'.format(y[pick_one]))


# 1.1.3 Plot 100 images
def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))  # sqrt features 400 to 20, size = 20
    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # randomly choose 100 from 5000
    sample_images = X[sample_idx, :]  # sample_images = 100*400

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),  # 10*r+c go over all 100 examples
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


# plot_100_image(X)
# plt.show()


# 1.1.4 Preprocess data X (5000,401)
# in the last 2 ex, X was read by panda, now its read by loadmat, and it's already a matrix (already array)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
# add x0=1 column in col 0, X is already a matrix, axis = 1 keep X as a matrix


# 1.1.5 Preprocess data y
# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# extend y from 5000*1 to 5000*10
y_matrix = []
for k in range(1, 11):  # 1-10
    y_matrix.append((y == k).astype(int))  # y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] one-hot coding
# see figure in file

# last one is k==10, it's digit 0, bring it to the first position
y_matrix = [y_matrix[-1]] + y_matrix[:-1]  # last element + elements except last one
y = np.array(y_matrix)  # array y for following use (TNC)


# y=10 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] column vector
# print(y.shape) # (10,5000)


# 1.2 train 1-d NN model------------------------------------------------------------------------
# 1d model means only classify into 2 parts, theta = (401,)
# can extend to multi-class by using one-vs-all
# 1.2.1 sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 1.2.2 cost function
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))  # can also use multiply
# y(5000,) X(5000,401) theta(401,1)


# 1.2.3 regularized_cost (ignore theta_0)
def regularized_cost(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]  # get rid of theta_0
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()  # leave theta 0 alone, lambda l=1
    # sum() sum up all theta in theta matrix except theta_0
    return cost(theta, X, y) + regularized_term


# 1.2.4 gradient
def gradient(theta, X, y):
    # just 1 batch gradient
    # X (5000,401)  X.T (401,5000) theta (401,1) y(5000,)
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


# 1.2.5 regularized gradient
def regularized_gradient(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])  # add theta_0 = 0 to theta  (401,1)

    return gradient(theta, X, y) + regularized_term  # return (401,1)


# 1.2.6 Logistic regression
def logistic_regression(X, y, l=1):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.zeros(X.shape[1])  # init theta (401,)

    # train it
    # minimize doesn't need args to be 1d array?
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    # get trained parameters
    final_theta = res.x  # optimal theta (401,)
    return final_theta


# 1.2.7 prediction
def predict(x, theta):
    prob = sigmoid(x @ theta)  # predict g(h_theta)  (5000,401)*(401,1) = (5000,1)
    return (prob >= 0.5).astype(int)  # one-hot coding result z (5000,)


# 1.3 Test -----------------------------------------------------------
# 1d nn, only classify into 2 types, use y[0] as y
t0 = logistic_regression(X, y[0])  # X(5000,401), y(10,5000), y[0] (5000,)
# t0  is the optimal theta (401,)
y_pred = predict(X, t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))
