# ex7 2D k means

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import numpy as np

# 1.1 Data preprocess----------------------------------------------------------------------
# 1.1.1 load
# mat = sio.loadmat('ex7data1.mat')
# data1 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
mat = sio.loadmat('ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])


# 1.1.2 plot
# sns.set(context="notebook", style="white")
# sns.lmplot('X1', 'X2', data=data1, fit_reg=False)
# sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
# plt.show()


# 1.2 K means---------------------------------------------------------------------------------
# 1.2.1 add centroids to data
def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)  # combine

    # calculate new centroids for each clusters
    return data_with_c.groupby('C', as_index=False).\
        mean(). \
        sort_values(by='C'). \
        drop('C', axis=1). \
        values


# 1.2.2 random init
def random_init(data, k):
    # k int
    return data.sample(k).values  # randomly choose k samples matrix from data set as init centroids (k,n)


# 1.2.3 find and assign clusters with a given example
def _find_your_cluster(x, centroids):
    # x(n,) centroids(k,n)
    distances = np.apply_along_axis(func1d=np.linalg.norm,  # this give you L2 norm
                                    axis=1,  # apply along row
                                    arr=centroids - x)  # calculate the distance of x from each centroids
    return np.argmin(distances)  # returns a int k that is closest to x


def assign_cluster(data, centroids):
    # assign each node in data a cluster
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centroids),
                               axis=1,  # apply along row (examples x1 x2 ...)
                               arr=data.values)  # take data values as a matrix


# 1.2.4 Cost
def cost(data, centroids, C):
    m = data.shape[0]
    expand_C_with_centroids = centroids[C]  # create a matrix contains all examples' clusters (m,n)

    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=data.values - expand_C_with_centroids)
    # calculate distance of each x from its C, and output as a matrix

    return distances.sum() / m  # return the mean of this matrix


# 1.2.5 iterations
def _k_means_iter(data, k, epoch=100, tol=0.0001):
    centroids = random_init(data, k)  # centroids (k,n)
    cost_progress = []

    for i in range(epoch):
        # print('running epoch {}'.format(i))
        C = assign_cluster(data, centroids)  # current clusters C (m,)
        centroids = new_centroids(data, C)  # compute new centroids for each clusters
        cost_progress.append(cost(data, centroids, C))  # compute cost for each iteration

        if len(cost_progress) > 1:  # early break if find last two cost is close (converge)
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break

    return C, centroids, cost_progress[-1]


def k_means(data, k, epoch=100, n_init=10):

    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)])  # try to init 10 times of 100 iterations

    least_cost_idx = np.argmin(tries[:, -1])  # find the index has smallest cost

    return tries[least_cost_idx]  # return the result of that time


# 1.3 Test-------------------------------------------------------------------------------
best_C, best_centroids, least_cost = k_means(data2, 3)
data_with_c = combine_data_C(data2, best_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
plt.show()