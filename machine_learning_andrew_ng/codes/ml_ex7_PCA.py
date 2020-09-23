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


def read_data(path):
    data = loadmat(path)
    data_x = data['X']
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


if __name__ == '__main__':
    X = read_data(path1)
    # plot_data(X)
    X_norm, means, stds = feature_normalize(X)  # 50,2
    U_out, S_out, V_out = pca(X_norm)  # 带入规范化值X_norm
    # plot_computed_eigen_vectors(X, means, S_out, U_out)
    plot_data(X_norm)
