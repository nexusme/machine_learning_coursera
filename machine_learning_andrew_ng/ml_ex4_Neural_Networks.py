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


# show 100 images
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
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 前馈
def forward_propagate(m_in, X, theta1, theta2):
    a1 = np.insert(X, 0, values=np.ones(m_in), axis=1)  # X0假设为1，为求出weights的第一个参数，加了一列1
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m_in), axis=1)  # 求a2
    z3 = a2 @ theta2.T
    h = sigmoid(z3)  # hθx 结果矩阵
    return a1, z2, a2, z3, h


# 对y标签进行编码 需要转为5000*10
def encode_y(y_in):
    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(y_in)
    return y_one_hot


def nnCostFunction(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)  # 对y标签进行编码 需要转为5000*10

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)  # 提取theta1 25 * 401

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)  # 提取theta2 10 * 26

    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)  # 使用前馈算法

    # compute the cost 根据公式求出cost
    J = 0
    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m
    return J


# 正则化cost function
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

    # add the cost regularization term 正则化项
    reg = (float(lambda_in) / (2 * m)) * (np.sum(np.power(theta_1[:, 1:], 2)) + np.sum(np.power(theta_2[:, 1:], 2)))
    return J + reg


# 随机初始theta 打破对称性
def rand_ini_para(hidden_size_in, input_size_in):
    # np.random.random(size) 返回size大小的0-1随机浮点数
    params = (np.random.random(
        size=hidden_size_in * (input_size_in + 1) + num_labels * (hidden_size_in + 1)) - 0.5) * 0.24
    return params


# 反向传播 返回正确的梯度下降结果
def back_prop(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    # “前向传递”计算整个神经网络的所有激活元，包括假设hΘ（x）的输出值
    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)

    # compute the cost
    J = 0
    delta1 = np.zeros(theta_1.shape)  # (25, 401)
    delta2 = np.zeros(theta_2.shape)  # (10, 26)

    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m

    # back prop 最小化J(Θ)
    # 对于层l中的每个节点j，我们要计算一个“误差项”δ（l)j，它测量出该节点对输出中的任何错误“负责”j的程度
    for t in range(m):
        # Set the input layer’s values (a(1) ) to the t-th training example x(t)
        a1t = np.mat(a_1[t, :])  # (1, 401)
        z2t = np.mat(z_2[t, :])  # (1, 25) computing the activations z(2)
        a2t = np.mat(a_2[t, :])  # (1, 26) a(2)
        ht = np.mat(h[t, :])  # (1, 10) z(3)
        yt = np.mat(y_fi[t, :])  # (1, 10) original

        d3t = ht - yt  # (1, 10) 算出error

        # 计算δ(2)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        gz = sigmoid_gradient(z2t)
        d2t = np.multiply((theta_2.T * d3t.T).T, gz)  # (1, 26)

        #  删除 δ(2)0
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m
    return J, delta1, delta2


# 反向传播正则化
def back_prop_reg(nn_params, input_size, hidden_size, label_num, X_in, y_in, lambda_in):
    m = len(X_in)

    y_fi = encode_y(y_in)

    theta_1 = nn_params[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)

    theta_2 = nn_params[hidden_size * (input_size + 1):].reshape(num_labels, hidden_size + 1)

    a_1, z_2, a_2, z_3, h = forward_propagate(m, X_in, theta_1, theta_2)

    # compute the cost
    J = 0
    delta1 = np.zeros(theta_1.shape)  # (25, 401)
    delta2 = np.zeros(theta_2.shape)  # (10, 26)

    for i in range(m):
        first_term = -y_fi[i, :] @ np.log(h[i, :])
        second_term = (1 - y_fi[i, :]) @ np.log(1 - h[i, :])
        J += np.sum(first_term - second_term)
    J = J / m
    reg = (float(lambda_in) / (2 * m)) * (np.sum(np.power(theta_1[:, 1:], 2)) + np.sum(np.power(theta_2[:, 1:], 2)))
    J = J+reg
    # back prop
    for t in range(m):
        # a_1 = np.mat(a_1)
        a1t = np.mat(a_1[t, :])  # (1, 401)
        z2t = np.mat(z_2[t, :])  # (1, 25)
        a2t = np.mat(a_2[t, :])  # (1, 26)
        ht = np.mat(h[t, :])  # (1, 10)
        yt = np.mat(y_fi[t, :])  # (1, 10)

        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)

        gz = sigmoid_gradient(z2t)
        d2t = np.multiply((theta_2.T * d3t.T).T, gz)  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta_1[:, 1:] * lambda_in) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta_2[:, 1:] * lambda_in) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad


def pre_accuracy(X, y, theta1, theta2):
    a1, z2, a2, z3, h = forward_propagate(len(X), X, theta1, theta2)
    y_pre = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pre, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))


if __name__ == "__main__":
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_t = 1

    load_theta_and_unroll(path2)

    data_X, data_y = load_data(path1)

    paras = rand_ini_para(hidden_layer_size, input_layer_size)

    # plot_100_images(data_X, data_y)
    nn_theta = load_theta_and_unroll(path2)

    # nnCostFunctionReg(nn_theta, input_layer_size, hidden_layer_size, num_labels, data_X, data_y, lambda_t)
    # sigmoid_gradient(0)
    # back_prop(nn_theta, input_layer_size, hidden_layer_size, num_labels, data_X, data_y, lambda_t)
    Cost, grad_result = back_prop_reg(nn_theta, input_layer_size, hidden_layer_size, num_labels, data_X, data_y,
                                      lambda_t)

    # minimize the objective function
    fmin = minimize(fun=back_prop_reg, x0=paras,
                    args=(input_layer_size, hidden_layer_size, num_labels, data_X, data_y, lambda_t),
                    method='TNC', jac=True, options={'maxiter': 250})

    theta_final_1 = np.mat(
        np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
    theta_final_2 = np.mat(
        np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))

    pre_accuracy(data_X, data_y, theta_final_1, theta_final_2)
