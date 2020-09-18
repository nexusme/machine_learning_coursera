import pandas as pd
import scipy
import sklearn
from sklearn import svm

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

path1 = 'ex6data1.mat'
path2 = 'ex6data2.mat'
path3 = 'ex6data3.mat'


def read_data(path):
    data = loadmat(path)
    read_X = data['X']  # 51,2
    read_y = data['y']  # 51,1
    return read_X, read_y


def read_x_y_val(path):
    data = loadmat(path)
    read_X = data['X']  # 51,2
    read_y = data['y']  # 51,1
    read_X_val = data['Xval']
    read_y_val = data['yval']
    return read_X, read_y, read_X_val, read_y_val


def data_to_df(x, y):
    save_list = []
    for i in range(len(x)):
        save_list.append([x[i][0], x[i][1], y[i][0]])
    df = pd.DataFrame(save_list, columns=['x1', 'x2', 'y'])
    return df


def plot_data(df_in):
    pos = df_in[df_in['y'] == 1]
    neg = df_in[df_in['y'] == 0]
    plt.plot(pos['x1'], pos['x2'], 'b+', markersize=4)
    plt.plot(neg['x1'], neg['x2'], 'ro', markersize=4)
    # plt.show()


def gaussian_kernel(x1, x2, sigma):
    result = np.exp(-np.power(x1 - x2, 2).sum() / (2 * sigma ** 2))
    print(result)
    return result


def plot_boundary(C, x_in, y_in):  # linear
    clf = svm.SVC(C=C, kernel='linear')  # 选择SVM的核函数 以及C
    model = clf.fit(x_in, y_in.flatten())  # 拟合
    # score = svc1.score(data_X, data_y.flatten()) # 准确率
    x = np.linspace(-0.5, 4.5, 500)
    y = np.linspace(1.3, 5, 500)
    xx, yy = np.meshgrid(x, y)
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)
    plt.show()


def plot_boundary_gaussian(x_in, y_in, sigma, C):
    sigma = sigma  # 0.1
    gamma = np.power(sigma, -2.) / 2
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)  # C=1
    model = clf.fit(x_in, y_in.flatten())
    x_min, x_max = x_in[:, 0].min() * 1.2, x_in[:, 0].max() * 1.1
    y_min, y_max = x_in[:, 1].min() * 1.1, x_in[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)
    plt.show()


def get_c_and_sig(x_in, y_in, x_val_in, y_val_in):
    C = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
    sig = C
    result = []

    for c in C:
        for sig_single in sig:
            gamma = np.power(sig_single, -2.) / 2  # add gamma
            model = svm.SVC(C=c, kernel='rbf', gamma=gamma)  # get model
            model.fit(x_in, y_in.flatten())  # train model
            result.append([c, sig_single, model.score(x_val_in, y_val_in)])  # get scores

    df = pd.DataFrame(result, columns=['c', 'sig', 'score'])
    new_df = df.sort_values('score', ascending=False).head(1)
    final_c = new_df['c'].values[0]
    final_sig = new_df['sig'].values[0]
    print(final_c, final_sig)
    return final_c, final_sig


if __name__ == '__main__':
    # data_X, data_y = read_data(path1)
    # data_df = data_to_df(data_X, data_y)

    # plot_data(data_df)  # linear svm model predict results Data set 1
    # plot_boundary(100, data_X, data_y)  # draw results

    # test1 = np.mat([1, 2, 1])
    # test2 = np.mat([0, 4, -1])
    # sig = 2
    # gaussian_kernel(test1, test2, sig)

    # data_X_2, data_y_2 = read_data(path2)
    # data_df_2 = data_to_df(data_X_2, data_y_2)
    # plot_data(data_df_2)  # plot data set 2
    # plot_boundary_gaussian(data_X_2, data_y_2)

    data_X_3, data_y_3, data_X_val, data_y_val = read_x_y_val(path3)
    data_df_3 = data_to_df(data_X_3, data_y_3)
    C_get, sig_get = get_c_and_sig(data_X_3, data_y_3, data_X_val, data_y_val)
    plot_data(data_df_3)  # plot data set 2
    plot_boundary_gaussian(data_X_3, data_y_3, sig_get, C_get)
