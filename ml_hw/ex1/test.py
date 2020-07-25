import numpy as np

a = np.random.randint(10, size=(97, 2))

b = np.random.randint(10, size=(2, 1))


# c = np.multiply(a, b)
# print(np.sum(c))
f = a @ b
print(f.shape)



