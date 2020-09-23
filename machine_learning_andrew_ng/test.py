"""
Created on Fri Jul  3 18:23:49 2020
@author: cheetah023
"""

import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
from skimage import io


# 函数定义
def findClosestCentroids(X, centroids):
    # 样本数量
    m, n = X.shape
    # 形心数量
    centroids = np.reshape(centroids, [len(centroids), n])
    K = centroids.shape[0]
    idx = np.zeros([m, 1])
    for i in range(0, m):
        distance_t = np.sum((X[i, :] - centroids[0, :]) ** 2)
        j_t = 0
        for j in range(1, K):
            distance = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if distance < distance_t:
                distance_t = distance
                j_t = j
        idx[i] = j_t
    return idx


def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros([K, n])
    for i in range(0, K):
        count = 0
        for j in range(0, m):
            if idx[j] == i:
                centroids[i, :] += X[j, :]
                count += 1
        centroids[i, :] /= count
    return centroids


def plotDataPoints(X, idx, K):
    m = X.shape[0]
    color = ['b', 'r', 'y', 'g', 'm', 'c', 'k', 'w']
    for i in range(0, m):
        plt.scatter(X[i, 0], X[i, 1], c=color[int(idx[i, 0])])


def plotProgresskMeans(X, centroids, idx, K):
    plotDataPoints(X, idx, K)
    for i in range(0, K):
        plt.plot(centroids[:, 0 + 2 * i], centroids[:, 1 + 2 * i], 'kx--', markersize=8)


def runkMeans(X, initial_centroids, max_iters, plot_progress):
    m, n = X.shape
    idx = np.zeros([m, 1])
    K = len(initial_centroids)
    initial_centroids = np.reshape(initial_centroids, [K, n])
    centroids = initial_centroids
    # 将每次迭代的形心点放在一行
    # 每次迭代centroids_history增加一行
    centroids_history = centroids.flatten()
    if plot_progress:
        plt.figure()

    for i in range(0, max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
        centroids_history = np.row_stack([centroids_history, centroids.flatten()])

    if plot_progress:
        plotProgresskMeans(X, centroids_history, idx, K)
    return centroids, idx


def kMeansInitCentroids(X, K):
    idx = np.random.choice(X.shape[0], K)
    centroids = X[idx, :]
    return centroids


# Part 1: Find Closest Centroids
data = sci.loadmat('ex7data2.mat')
# print('data.keys',data.keys())
X = data['X']
print('X:', X.shape)
K = 3
initial_centroids = [[3, 3], [6, 2], [8, 5]]

idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 examples:', idx[0:3] + 1)
print('(the closest centroids should be 1, 3, 2 respectively)')

# Part 2: Compute Means
centroids = computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('(the centroids should be')
print('[ 2.428301 3.157924 ]')
print('[ 5.813503 2.633656 ]')
print('[ 7.119387 3.616684 ]')

# Part 3: K-Means Clustering
K = 3
max_iters = 10
initial_centroids = [[3, 3], [6, 2], [8, 5]]
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, True)

# Part 4: K-Means Clustering on Pixels
data = io.imread('bird_small.png')
print('data:', data.shape)
m, n, k = data.shape
data = data / 255
X = np.reshape(data, [-1, 3])
K = 16;
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, False)

# Part 5: Image Compression
idx = findClosestCentroids(X, centroids)
X_recovered = np.zeros_like(X)
for i in range(0, m * n):
    X_recovered[i] = centroids[int(idx[i, 0])]
X_recovered = X_recovered.reshape([m, n, k])

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(data)
axes[0].set_title('original')
axes[1].imshow(X_recovered)
axes[1].set_title('Compressed, with %d colors.'.format(K))
plt.show()