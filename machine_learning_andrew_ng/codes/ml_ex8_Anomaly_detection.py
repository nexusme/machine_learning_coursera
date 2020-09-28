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

path1 = '../files/ex8data1.mat'
path2 = '../files/ex8data2.mat'


def read_data(path):
    data = loadmat(path)
    data_X = data['X']
    data_X_val = data['Xval']
    data_y_val = data['yval']
    return data_X, data_X_val, data_y_val


def plot_data(data):
    plt.plot(data[:, 0], data[:, 1], 'bx', markersize='3')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    # plt.show()


def estimate_gaussian(data):
    m, n = data.shape  # 307*2
    mu = sum(data) / m
    sig_power_2 = 1 / m * sum(np.power(data - mu, 2))
    for i in range(0, n):
        sig_power_2[i] = np.sum((data[:, i] - mu[i]) ** 2) / m
    return mu, sig_power_2


# 也可以使用多元高斯分布模型计算概率
def multi_gaussian(x, mu, sigma):
    p = np.zeros((len(x), 1))
    n = len(mu)
    if np.ndim(sigma) == 1:
        sigma = np.diag(sigma)
    for i in range(len(x)):
        p[i] = (2 * np.pi) ** (-n / 2) * np.linalg.det(sigma) ** (-1 / 2) * np.exp(
            -0.5 * (x[i, :] - mu).T @ np.linalg.inv(sigma) @ (x[i, :] - mu))
    return p


def visualize_fit(X, mu, sigma2):
    plot_data(X)
    x1 = np.linspace(0, 30)
    x2 = np.linspace(0, 30)
    xx, yy = np.meshgrid(x1, x2)
    X_temp = np.column_stack([xx.flatten(), yy.flatten()])
    print('X_temp:', X_temp.shape)
    p = multi_gaussian(X_temp, mu, sigma2)
    p = np.reshape(p, xx.shape)
    levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, p, levels)
    plt.tight_layout()
    # plt.show()


def select_threshold(yval, pval):
    best_epsilon = 0
    best_F1 = 0
    epsilons = np.linspace(min(pval), max(pval), 1000)

    for epsilon in epsilons:
        cv_predictions = (pval < epsilon)
        fp = np.sum((cv_predictions == 1) & (yval == 0))
        tp = sum((cv_predictions == 1) & (yval == 1))
        fn = sum((cv_predictions == 0) & (yval == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_F1:
            best_epsilon = epsilon
            best_F1 = f1
    return best_epsilon, best_F1


def draw_outliers(XX_in, outliers_in, mu_in, sig_in):
    visualize_fit(XX_in, mu_in, sig_in)
    plt.scatter(XX_in[outliers_in[0], 0], XX_in[outliers_in[0], 1], marker='o', edgecolor='r')
    plt.show()


if __name__ == '__main__':
    XX, X_val, y_val = read_data(path1)
    # plot_data(X)
    # mu_out, sig_out = estimate_gaussian(XX)
    # P = multi_gaussian(XX, mu_out, sig_out)
    # visualize_fit(XX, mu_out, sig_out) # Visualize the data set and its estimated distribution.
    # p_val = multi_gaussian(X_val, mu_out, sig_out)  # 用给定的交叉验证集概率找到一个好的epsilon阈值
    # epsilon_out, F1 = select_threshold(y_val, p_val)  # 得到最好的esp和f1 score
    # outliers = np.where(P < epsilon_out)
    # draw_outliers(XX, outliers, mu_out, sig_out)

    # High dimensional data set
    XX_h, X_val_h, y_val_h = read_data(path2)
    mu_h, sig_h = estimate_gaussian(XX_h)
    P_h = multi_gaussian(XX_h, mu_h, sig_h)
    p_val_h = multi_gaussian(X_val_h, mu_h, sig_h)
    ep_h, f1_h = select_threshold(y_val_h, p_val_h)
    print(ep_h)
    print(f1_h)
