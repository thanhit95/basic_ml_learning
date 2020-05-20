import numpy as np
import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from display_network import *
import imageio

mnist.temporary_dir = lambda: './mnist/'


def load_dataset(N_limit=1000, normalization=True):
    X = mnist.test_images()

    nsamples, ny, nx = X.shape
    X = X.reshape((nsamples, ny * nx))

    X = np.asarray(X)[:N_limit, :]

    if normalization:
        X = X / 256.0

    return X, N_limit


######################################
#                MAIN
######################################

X, _ = load_dataset()
K = 10

kmeans = KMeans(n_clusters=K).fit(X)
print('cluster_centers:', kmeans.cluster_centers_.T.shape)

pred_label = kmeans.predict(X)
# print(pred_label.shape)
# print(pred_label)


A = display_network(kmeans.cluster_centers_.T, K, 1)

f1 = plt.imshow(A, interpolation='nearest', cmap='jet')
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.savefig('a1.png', bbox_inches='tight')
plt.show()


N0 = 20
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
    Xk = X[pred_label == k, :]

    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(N0).fit(Xk)
    dist, nearest_id = neigh.kneighbors(center_k, N0)

    X1[N0*k: N0*k + N0, :] = Xk[nearest_id, :]
    X2[N0*k: N0*k + N0, :] = Xk[:N0, :]


plt.axis('off')
A = display_network(X2.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest')
plt.gray()
plt.show()

# import scipy.misc
# scipy.misc.imsave('bb.png', A)


# plt.axis('off')
# A = display_network(X1.T, 10, N0)
# scipy.misc.imsave('cc.png', A)
# f2 = plt.imshow(A, interpolation='nearest' )
# plt.gray()

# plt.show()
