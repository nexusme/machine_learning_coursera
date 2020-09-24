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

path1 = '../files/ex7data1.mat'
path2 = '../files/ex7faces.mat'


def read_data(path):
    data = loadmat(path)
    data_x = data['X']
    print(data_x.shape)
    return data_x


def plot_data(data):
    # print(data[:, 0])
    plt.scatter(data[:, 0], data[:, 1], marker='o', c='', edgecolors='b')
    plt.show()


def feature_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    data = (data - mean) / std
    return data, mean, std


def pca(data):  # 计算data的协方差矩阵的特征向量
    # U = np.zeros(data.shape[1])
    # S = np.zeros(data.shape[1])
    data_mat = np.mat(data)
    sigma = (data_mat.T @ data_mat) / len(data)
    U, S, V = np.linalg.svd(sigma)  # 特征向量U, 特征值（对角)S
    return U, S, V


def plot_computed_eigen_vectors(X_in, mean, s_in, u_in):  # 画出协方差
    plt.figure(figsize=(7, 5))
    plt.scatter(X_in[:, 0], X_in[:, 1], facecolors='none', edgecolors='b')

    plt.plot([mean[0], mean[0] + 1.5 * s_in[0] * u_in[0, 0]],
             [mean[1], mean[1] + 1.5 * s_in[0] * u_in[0, 1]],
             c='r', linewidth=3, label='First Principal Component')
    plt.plot([mean[0], mean[0] + 1.5 * s_in[1] * u_in[1, 0]],
             [mean[1], mean[1] + 1.5 * s_in[1] * u_in[1, 1]],
             c='g', linewidth=3, label='Second Principal Component')
    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.show()


# Projecting the data onto the principal components
def project_data(X_norm_in, U_in, K_in):
    z = X_norm_in @ U_in[:, :K_in]
    return z


# Recovers an approximation of the original data when using the projected data
def recover_data(Z_in, U_in, K_in):
    recovered = Z_in @ U_in[:, :K_in].T
    return recovered


# Visualizing the projections
def plot_projections(X_norm_in, X_rec_in):
    X_rec_in = np.array(X_rec_in)
    plt.figure(figsize=(7, 5))
    plt.axis("equal")
    plt.scatter(X_norm_in[:, 0], X_norm_in[:, 1], s=30, facecolors='none',
                edgecolors='b', label='Original Data Points')
    plt.scatter(X_rec_in[:, 0], X_rec_in[:, 1], s=30, facecolors='none',
                edgecolors='r', label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
    plt.xlabel('x1 [Feature Normalized]', fontsize=14)
    plt.ylabel('x2 [Feature Normalized]', fontsize=14)
    plt.grid(True)

    for x in range(X_norm_in.shape[0]):
        plt.plot([X_norm_in[x, 0], X_rec_in[x, 0]], [X_norm_in[x, 1], X_rec_in[x, 1]], 'k--')
        # 输入第一项全是X坐标，第二项都是Y坐标
    plt.legend()
    plt.show()


def display_face(X_in, row, col):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(X_in[r * col + c].reshape(32, 32).T, cmap='Greys_r')
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
    plt.show()


if __name__ == '__main__':
    X = read_data(path1)
    # plot_data(X)
    X_norm, means, stds = feature_normalize(X)  # 50,2
    U_out, S_out, V_out = pca(X_norm)  # 带入规范化值X_norm
    # plot_computed_eigen_vectors(X, means, S_out, U_out)
    # plot_data(X_norm)
    Z = project_data(X_norm, U_out, 1)
    X_rec = recover_data(Z, U_out, 1)
    # plot_projections(X_norm, X_rec)

    data_face = read_data(path2)

    # display_face(data_face, 10, 10) # display faces

    # PCA
    X_norm_face, means_face, stds_face = feature_normalize(data_face)
    U_face, S_face, V_face = pca(X_norm_face)
    Z_face = project_data(X_norm_face, U_face, 36)  # projection
    X_rec_face = recover_data(Z_face, U_face, 36)  # recover data
    display_face(X_rec_face, 10, 10)  # plot faces
