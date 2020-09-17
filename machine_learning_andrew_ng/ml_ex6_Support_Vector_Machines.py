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


def read_data(path):
    data = loadmat(path)
    read_X = data['X']  # 51,2
    read_y = data['y']  # 51,1
    return read_X, read_y


def data_to_df(x, y):
    save_list = []
    for i in range(len(x)):
        save_list.append([x[i][0], x[i][1], y[i][0]])
    df = pd.DataFrame(save_list, columns=['x1', 'x2', 'y'])
    return df


def plot_data(df_in):
    pos = df_in[df_in['y'] == 1]
    neg = df_in[df_in['y'] == 0]
    plt.plot(pos['x1'], pos['x2'], 'b+')
    plt.plot(neg['x1'], neg['x2'], 'ro')
    plot_boundary(100, data_X, data_y)

    plt.show()


def plot_boundary(C, x_in, y_in):
    model = svm.SVC(C=C, kernel='linear')  # 选择SVM的核函数 以及C
    clf = model.fit(x_in, y_in.flatten())  # 拟合
    # score = svc1.score(data_X, data_y.flatten()) # 准确率
    x = np.linspace(-0.5, 4.5, 500)
    y = np.linspace(1.3, 5, 500)
    xx, yy = np.meshgrid(x, y)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)


if __name__ == '__main__':
    data_X, data_y = read_data(path1)
    data_df = data_to_df(data_X, data_y)
    # linear svm model predict results
    plot_data(data_df)
