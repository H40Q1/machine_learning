# ex1 normal equation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3.1 Dealing with data----------------------------------------------------------
# 3.1.1 read file
path = "ex1data1.txt"

# 3.1.2 write as a table
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()  # print(data.head())
data.describe()  # print(data.describe())

# 3.1.3 plot data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))  # plt.show()

# 3.1.4 Preprocess data
data = (data - data.mean()) / data.std()  # mean normalization
data.insert(0, 'Ones', 1)  # add x0=1 column, 0=column no, 1=val added, 'Ones' is header
cols = data.shape[1]  # get number of columns, which is n+2
X = data.iloc[:, 0:cols - 1]  # X is the all the columns except the last one
y = data.iloc[:, cols - 1:cols]  # Y is the last column
X = np.matrix(X.values)  # convert X and Y from csv table to numpy matrix, every xi given is (xi)T
y = np.matrix(y.values)


# 3.2 Normal Equation
# 3.2.1 NE function
def normalEquation(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # np.linalg
    return theta


# 3.2.2 result
final_theta = normalEquation(X, y)
print(final_theta)
