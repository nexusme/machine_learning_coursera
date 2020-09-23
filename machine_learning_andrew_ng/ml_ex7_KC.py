import pandas as pd
import scipy
import sklearn
from skimage import io
from sklearn import svm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

path1 = 'ex7data2.mat'
path2 = 'bird_small.png'


def read_data(path):
    data = loadmat(path)
    read_X = data['X']  # 300,2
    return read_X


# Part 4: K-Means Clustering on Pixels

def on_pixles(path):
    data = io.imread(path)
    # print('data:', data.shape)
    mm, nn, kk = data.shape
    print(mm, nn, kk)
    data = data / 255
    X = np.reshape(data, [mm * nn, 3])
    return data, X


def find_closest_centroids(x_in, cent):
    m, n = x_in.shape
    # 形心数量
    centroids = np.reshape(cent, [len(cent), n])
    K = centroids.shape[0]
    idx = np.zeros([m, 1])
    for i in range(0, m):
        distance_t = np.sum(np.power((x_in[i, :] - centroids[0, :]), 2))
        j_t = 0
        for j in range(1, K):
            distance = np.sum(np.power((x_in[i, :] - centroids[j, :]), 2))
            if distance < distance_t:
                distance_t = distance
                j_t = j
        idx[i] = j_t
    return idx


def compute_centroids(X_in, idx_in, K_in):
    mm, nn = X_in.shape
    centroids = np.zeros([K_in, nn])
    for ii in range(0, K_in):
        count = 0
        for j in range(0, mm):
            if idx_in[j] == ii:
                centroids[ii, :] += X_in[j, :]
                count += 1
        centroids[ii, :] /= count
    return centroids


def plot_data_points(X, idx_in, K):
    m = X.shape[0]
    color = ['b', 'r', 'y', 'g', 'm', 'c', 'k', 'w']
    for i in range(0, m):
        plt.scatter(X[i, 0], X[i, 1], c='', edgecolors=color[int(idx_in[i, 0])])


def plot_progress_k_means(X, centroids_in, idx_in, K):
    plot_data_points(X, idx_in, K)
    for i in range(0, K):
        plt.plot(centroids_in[:, 0 + 2 * i], centroids_in[:, 1 + 2 * i], 'kx--', markersize=8)


def k_means_init_centroids(X_in, K_in):
    # cent = np.zeros([K_in, len(X_in)])
    index_list = [i for i in range(len(X_in))]
    random.shuffle(index_list)  # Randomly reorder the indices of examples
    # print(index_list)
    cent = X_in[index_list[0:K_in], :]  # Take the first K examples as centroids
    # print(cent)
    return cent


def run_k_means(X, initial_centroids, max_iters, plot_progress):
    m, n = X.shape  # n is 3
    id_x = np.zeros([m, 1])
    K = len(initial_centroids)  # k is 3
    initial_centroids = np.reshape(initial_centroids, [K, n])
    centroid = initial_centroids
    # 将每次迭代的形心点放在一行
    # 每次迭代centroids_history增加一行
    centroids_history = centroid.flatten()
    if plot_progress:
        plt.figure()

    for i in range(0, max_iters):
        id_x = find_closest_centroids(X, centroid)
        centroid = compute_centroids(X, id_x, K)
        centroids_history = np.row_stack([centroids_history, centroid.flatten()])
    if plot_progress:
        plot_progress_k_means(X, centroids_history, id_x, K)
    plt.show()
    return centroid, id_x


if __name__ == '__main__':
    k = 3  # 3 Centroids
    max_iter = 10
    ini_centroids = np.mat([[3, 3], [6, 2], [8, 5]])  # Select an initial set of centroids size 3,2
    raw_data = read_data(path1)

    # Find the closest centroids for the examples using the initial_centroids
    # idx_value = find_closest_centroids(raw_data, ini_centroids)  # test
    # centroids_out = compute_centroids(raw_data, idx_value, k)  # test
    # [centroid_final, idx_final] = run_k_means(raw_data, ini_centroids, max_iter, True)  # K-means on example data set

    og_data, raw_data_2 = on_pixles(path2)
    ini_cent_2 = k_means_init_centroids(raw_data_2, k)
    [centroid_final, idx_final] = run_k_means(raw_data_2, ini_cent_2, max_iter, False)  # K-means on example data set

    # Image Compression
    idx_f = find_closest_centroids(raw_data_2, centroid_final)
    X_recovered = np.zeros_like(raw_data_2)
    m1, n1, K1 = 128, 128, 3
    print(m1, n1, K1)
    for i in range(0, m1 * n1):
        X_recovered[i] = centroid_final[int(idx_f[i, 0])]
    X_recovered = X_recovered.reshape([m1, n1, K1])

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(og_data)
    axes[0].set_title('original')
    axes[1].imshow(X_recovered)
    axes[1].set_title('Compressed,' + str(K1) + ' colors.')
    plt.show()
