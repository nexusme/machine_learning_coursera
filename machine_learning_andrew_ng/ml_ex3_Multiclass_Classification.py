import scipy
import sklearn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize

path1 = 'ex3data1.mat'


#  load mat data
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


# plot a random number
def plot_one_image(X, y):
    random_one = np.random.randint(0, 5000)  # choose row randomly
    image = X[random_one, :]  # fetch 400 pixels
    fig, ax = plt.subplots(figsize=(1, 1))  # one row one column
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')  # reshape image to 20*20's matrix
    plt.show()
    print('this number is {}'.format(y[random_one]))  # print value


def plot_100_images(X, y):
    random_100 = np.random.choice(np.arange(X.shape[0]), 100)  # choose 100 rows randomly
    images = X[random_100, :]  # (100,400)
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):  # reshape image to 20*20's matrix
            ax_array[row, column].matshow(images[10 * row + column].reshape((20, 20)),
                                          cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('these numbers are {}'.format(y[random_100]))  # print value


# Compute the sigmoid
def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def lr_costFunction(theta_in, X_in, y_in, lambda_in):
    # m = len(y_in)
    # y_in = np.mat(y_in)
    g = sigmoid(X_in @ theta_in)
    # J = 1 / m * (-y_in.T * np.log(g) - (1 - y_in.T) * np.log(1 - g)) + lambda_in / 2 / m * sum(
    #     np.power(theta_in[1:], 2))
    # print(J)
    # return J
    thetaReg = theta_in[1:]
    first = (-y_in.T * np.log(g).T) + (y_in.T - 1) * np.log(1 - g.T)
    reg = (thetaReg @ thetaReg) * lambda_in / (2 * len(X_in))
    return np.mean(first) + reg


def lr_gradient(theta_in, X_in, y_in, lambda_in):
    # m = len(y_in)
    # print(theta_in.shape)
    g = sigmoid(X_in @ theta_in)
    # grad = np.zeros([theta_in.shape[0], 1])
    # y_in = np.mat(y_in).T
    # # print(grad)
    # print(X_in[:, 0].T * (g - y_in))
    # grad[0:] = 1 / m * X_in[:, 0].T * (g - y_in)
    # grad[1:] = 1 / m * X_in[:, 1:].T * (g - y_in) + lambda_in / m * theta_in[1:]
    # # print(grad)
    # return grad
    thetaReg = theta_in[1:]
    first = 1 / len(X_in) * X_in.T @ (g - y_in).T
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (lambda_in / len(X_in)) * thetaReg])
    return first + reg


def one_vs_all(X_in, y_in, lambda_in, num_labels):
    n = X_in.shape[1]  # n (5000,401)
    theta_all = np.zeros((num_labels, n))
    # 10个分类器，每个分类器进行一次minimize
    for i in range(1, num_labels + 1):
        theta_i = np.zeros(n)
        y_i = np.array([1 if label == i else 0 for label in y_in])
        res = minimize(fun=lr_costFunction, x0=theta_i, args=(X_in, y_i, lambda_in), method='TNC',  # 返回一组符合十种数字的参数
                       jac=lr_gradient, options={'disp': True})
        # res = minimize(lr_costFunction, theta_i, args=(X_in, y_i, lambda_in), method='TNC', jac=lr_gradient)
        theta_all[i - 1] = res.x
    return theta_all


def predict(X_in, theta_in):
    y_p = sigmoid(X_in @ theta_in.T)  # 5000*10
    # 返回每一行最大概率值的索引
    y_pre = np.argmax(y_p, axis=1)
    return y_pre + 1


if __name__ == '__main__':
    data_x, data_y = load_data(path1)
    # plot_one_image(data_x, data_y)
    # plot_100_images(data_x, data_y)
    # theta_t = np.mat([-2, -1, 1, 2]).T
    # X_t = np.mat([([1, num / 10, (num + 5) / 10, (num + 10) / 10]) for num in range(1, 6)])
    # y_t = np.mat([num if num >= 0.5 else num for num in [1, 0, 1, 0, 1]]).T
    # lambda_t = 3
    XX = np.insert(data_x, 0, 1, axis=1)
    yy = data_y.flatten()
    # lr_costFunction(theta_t, X_t, y_t, lambda_t)
    # lr_gradient(theta_t, X_t, y_t, lambda_t)
    theta_final = one_vs_all(XX, yy, 1, 10)
    pre_result = predict(XX, theta_final)
    print(pre_result)
    print(yy.shape)
    print(str(np.mean(yy == pre_result) * 100) + '%')
