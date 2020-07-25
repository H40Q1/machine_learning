# ex 7 k means using sklearn

import pandas as pd
import scipy.io as sio
from sklearn.cluster import KMeans


# load
mat = sio.loadmat('ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])

# create model
sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data2)

# output prediction
sk_C = sk_kmeans.predict(data2)


