import scipy
import sklearn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

path1 = 'ex3weights.mat'
path2 = 'ex3data1.mat'


#  load mat data
def load_theta(path):
    data = loadmat(path)
    t1 = data['Theta1']
    t2 = data['Theta2']
    return t1, t2


#  load mat data
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


# plot a random number
def plot_one_image(num, X, y):
    image = X[num, :]  # fetch 400 pixels
    fig, ax = plt.subplots(figsize=(1, 1))  # one row one column
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')  # reshape image to 20*20's matrix
    plt.show()
    print('this number is {}'.format(y[num]))  # print value


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


def predict(X_in, theta_in, theta_in_2):
    y_p = sigmoid(X_in @ theta_in.T)  # 5000*10
    y_p_insert = np.insert(y_p, 0, 1, axis=1)
    y_pp = sigmoid(y_p_insert @ theta_in_2.T)
    # 返回每一行最大概率值的索引
    y_pre = np.argmax(y_pp, axis=1) + 1

    return y_pre

#
# def predict_example(m_in, X, y, Theta1, Theta2):
#     rp = np.random.permutation(m_in)
#     for i in range(m_in):
#         plot_one_image(rp[i], X, y)
#         X_manage = X[rp[i], :]
#         X_manage = np.insert(X_manage, 0, 1, axis=1)
#         pre = predict(X_manage, Theta1, Theta2)
#
#         print('\nNeural Network Prediction: %d (digit %d)\n', pre, pre % 10)
#     # print(rp)


if __name__ == '__main__':
    # 20x20 Input Images of Digits
    input_layer_size = 400
    # 25 hidden units
    hidden_layer_size = 25
    # 10 labels, from 1 to 10, 10 is 0
    num_labels = 10
    # length of x
    m = 5000

    theta_1, theta_2 = load_theta(path1)
    data_x, data_y = load_data(path2)
    data_x_insert_1 = np.insert(data_x, 0, 1, axis=1)

    # plot_100_images(data_x, data_y)
    pre_result = predict(data_x_insert_1, theta_1, theta_2)
    # print(str(np.mean(data_y.flatten() == pre_result) * 100) + '%')
    # predict_example(m, data_x, data_y, theta_1, theta_2)
