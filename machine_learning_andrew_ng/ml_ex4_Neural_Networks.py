import scipy
import sklearn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

from sklearn.preprocessing import OneHotEncoder

path1 = 'ex4data1.mat'

path2 = 'ex4weights.mat'


#  load mat data and unroll
def load_theta_and_unroll(path):
    data = loadmat(path)
    # t1 = list(data['Theta1'].flatten())
    # t2 = list(data['Theta2'].flatten())
    tt = np.array(list(data['Theta1'].flatten()) + list(data['Theta2'].flatten())).flatten()
    # print(tt.shape)
    return tt


#  load mat data
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


def plot_100_images(X_in, y_in):
    random_100 = np.random.permutation(X_in.shape[0])[:100]  # choose 100 rows
    images = X_in[random_100, :]  # (100,400)
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):  # reshape image to 20*20's matrix
            ax_array[row, column].matshow(images[10 * row + column].reshape((20, 20)),
                                          cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('these numbers are {}'.format(y_in[random_100]))  # print value


# Compute the sigmoid
def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def sigmoid_gradient(z):
    print(sigmoid(z) * (1 - sigmoid(z)))
    return sigmoid(z) * (1 - sigmoid(z))


def forward_propagate(m_in, X, theta1, theta2):
    a1 = np.insert(X, 0, values=np.ones(m_in), axis=1)
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m_in), axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


# 对y标签进行编码 需要转为5000*10
def encode_y(y_in):
    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(y_in)
    return y_one_hot


def nnCostFunction(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m
    print(J)
    return J


# 正则化
def nnCostFunctionReg(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m

    # add the cost regularization term
    reg = (float(lambda_in) / (2 * m)) * (np.sum(np.power(theta_1[:, 1:], 2)) + np.sum(np.power(theta_2[:, 1:], 2)))
    print(J + reg)
    return J + reg


# 随机初始theta
def rand_ini_para(hidden_size_in, input_size_in):
    # np.random.random(size) 返回size大小的0-1随机浮点数
    params = (np.random.random(
        size=hidden_size_in * (input_size_in + 1) + num_labels * (hidden_size_in + 1)) - 0.5) * 0.24
    return params


# 反向传播
def back_prop(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m

    print(J)

    return J


if __name__ == "__main__":
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_t = 1

    load_theta_and_unroll(path2)

    data_X, data_y = load_data(path1)
    # plot_100_images(data_X, data_y)
    nn_theta = load_theta_and_unroll(path2)

    # nnCostFunctionReg(nn_theta, input_layer_size, hidden_layer_size, num_labels, data_X, data_y, lambda_t)

    # sigmoid_gradient(0)
