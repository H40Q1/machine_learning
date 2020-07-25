# ex7 using k means to compress image
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans


# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('bird_small.png') / 255.  # pic.shape （128, 128, 3)


# serialize data
data = pic.reshape(128*128, 3)  # (16384.3)


# K means
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)  # k = 16
model.fit(data)
centroids = model.cluster_centers_  # (16,3) optimal centroids
C = model.predict(data)   # (16384,)  # clusters assigned to each examples
# centroids[C].shape (16384,3)


# create new pic with current clusters
compressed_pic = centroids[C].reshape((128, 128, 3))


# imshow
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
