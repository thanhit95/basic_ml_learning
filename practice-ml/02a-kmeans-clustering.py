import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(11)
plot_chars = ['b^', 'go', 'rs', 'k+', 'c*', 'yp', 'mx']


def kmeans_display(X, label, figure_id):
    K = np.amax(label) + 1
    print(K)
    plt.figure(figure_id)

    for i in range(K):
        Xi = X[label == i, :]
        plt.plot(Xi[:, 0], Xi[:, 1], plot_chars[i], markersize=4, alpha=.7)

    plt.axis('equal')
    # plt.show(block=False)


def kmeans_init_centers(X, K):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], K, replace=False)]


def kmeans_assign_labels(X, centers):
    # calculate pairwise distances between data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)


def kmeans_update_centers(X, K, labels):
    centers = np.zeros((K, X.shape[1]))

    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)

    return centers


def has_converged(old_centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in old_centers]) == set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    lst_centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0

    while True:
        labels.append(kmeans_assign_labels(X, lst_centers[-1]))
        new_centers = kmeans_update_centers(X, K, labels[-1])

        if has_converged(lst_centers[-1], new_centers):
            break
        
        lst_centers.append(new_centers)
        it += 1

    return (lst_centers, labels, it)


#################################################
#                    MAIN
#################################################

# INIT DATASET
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 700

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


kmeans_display(X, original_label, 1)


# PROCESS
(lst_centers, labels, it) = kmeans(X, K)

print('Centers found by our algorithm:')
print(lst_centers[-1])

kmeans_display(X, labels[-1], 2)

plt.show()
