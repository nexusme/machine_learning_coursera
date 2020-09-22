
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


def read_image(path):
    data = io.imread(path)
    print(data)


def find_closest_centroids(x_in, cent):
    idx = np.zeros([len(x_in), 1])
    err_list = []
    for i in range(len(x_in)):
        one_row = np.mat([[x_in[i][0], x_in[i][1]], [x_in[i][0], x_in[i][1]], [x_in[i][0], x_in[i][1]]])
        err_list.append([np.sum(row[0]) for row in np.power(one_row - cent, 2)])
    for i in range(len(err_list)):
        idx[i] = err_list[i].index(min(err_list[i])) + 1
    return idx


# Compute means based on the closest centroids found in the previous part.
def compute_centroids(X_in, idx_in, K_in):
    list_pre = [[idx_in.tolist()[i][0], X_in[i][0], X_in[i][1]] for i in range(len(X_in))]
    # centroids = np.mat([cc_df(list_pre, i) for i in range(1, k + 1)])
    # print(centroids)
    return np.mat([cc_df(list_pre, i) for i in range(1, K_in + 1)])


def cc_df(list_in, type_value):
    df = pd.DataFrame(list_in, columns=['type', 'x1', 'x2'])
    df_type = df[df['type'] == type_value].sum()
    value = df_type['type'] / type_value
    df_type = df_type / value
    return [df_type['x1'], df_type['x2']]


def plot_data_points(X, idx_in, K):
    m = X.shape[0]
    color = ['b', 'r', 'y', 'g', 'm', 'c', 'k', 'w']
    for i in range(0, m):
        plt.scatter(X[i, 0], X[i, 1], c='', edgecolors=color[int(idx_in[i, 0])])


def plot_progress_k_means(X, centroids_in, idx_in, K):
    plot_data_points(X, idx_in, K)
    for i in range(0, K):
        plt.plot(centroids_in[:, 0 + 2 * i], centroids_in[:, 1 + 2 * i], 'kx--', markersize=8)


def run_k_means(X, initial_centroids, max_iters, plot_progress):
    m, n = X.shape
    id_x = np.zeros([m, 1])
    K = len(initial_centroids)
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
    # idx_value = find_closest_centroids(raw_data, ini_centroids) # test
    # centroids_out = compute_centroids(raw_data, idx_value, k) # test
    # [centroid_final, idx_final] = run_k_means(raw_data, ini_centroids, max_iter, True)  # K-means on example dataset
    read_image(path2)
