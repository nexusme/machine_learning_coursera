import scipy
import sklearn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

path1 = '../files/ex5data1.mat'


#  load mat data
def load_data(path):
    data = loadmat(path)
    X_out = data['X']
    y_out = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']

    return X_out, y_out, Xval, yval, Xtest, ytest


def draw_plot(data_x, data_y):
    plt.plot(data_x, data_y, 'rx', 10)
    # Set the y−axis label
    plt.ylabel('Water flowing out of the dam (y)')
    # Set the x−axis label
    plt.xlabel('Change in water level (x)')
    # show plot
    plt.show()


def linear_reg_cost(theta_in, X_in, y_in, lambda_in):
    m = len(X_in)
    theta_in = theta_in.reshape(2, 1)
    J = 1 / (m * 2) * np.power(X_in @ theta_in - y_in, 2).sum()
    reg = lambda_in / (2 * m) * np.power((theta_in[1:] @ theta_in[1:].T), 2).sum()

    return J + reg


def linear_reg_gradient(theta_in, X_in, y_in, lambda_in):
    m = len(X_in)
    theta_in = theta_in.reshape(2, 1)
    grad = 1 / m * X_in.T @ (X_in @ theta_in - y_in) + (lambda_in / m) * theta_in

    return grad


def train_linear_reg(X_in, y_in, lambda_in):
    ini_theta = np.zeros([X_in.shape[1], 1])
    res = minimize(fun=linear_reg_cost, x0=ini_theta, args=(X_in, y_in, lambda_in), method='TNC',
                   jac=linear_reg_gradient, options={'maxiter': 200})
    # print(res.x)
    return res.x


def plot_over_fit_data(X_in, X_insert, y_in, theta_in):
    plt.plot(X_in, y_in, 'rx', 10)
    # Set the y−axis label
    plt.ylabel('Water flowing out of the dam (y)')
    # Set the x−axis label
    plt.xlabel('Change in water level (x)')
    # show plot
    plt.plot(X_in, X_insert @ theta_in, '-', 'LineWidth', 2)
    plt.show()


def learning_curve(X_in, y_in, X_val_in, y_val_in, lam_in):
    x = range(1, X_in.shape[0] + 1)
    training_costs = []
    val_costs = []
    for i in x:
        theta_s = train_linear_reg(X_in[:i, :], y_in[:i, :], lam_in)
        training_c = linear_reg_cost(theta_s, X_in[:i, :], y_in[:i, :], lam_in)
        val_c = linear_reg_cost(theta_s, X_val_in, y_val_in, lam_in)
        training_costs.append(training_c)
        val_costs.append(val_c)

    plt.plot(x, training_costs, label='trainging cost')
    plt.plot(x, val_costs, label='val cost')
    plt.legend()
    plt.xlabel('number of training examples')
    plt.ylabel('error')
    plt.show()


def poly_features(X_in, p_num):
    for i in range(2, p_num + 1):
        X_in = np.insert(X_in, X_in.shape[1], np.power(X_in[:, 1], p_num), axis=1)
    return X_in


if __name__ == "__main__":
    X, y, X_val, y_val, X_test, y_test = load_data(path1)
    # print(X.shape)  # 12,1
    # draw_plot(X, y)
    theta_test = np.mat([1, 1]).T
    lambda_test = 0
    X_in_cost = np.insert(X, 0, 1, axis=1)
    X_val_insert = np.insert(X_val, 0, 1, axis=1)

    # linear_reg_cost(X_in_cost, y, theta_test, lambda_test)
    final_theta = train_linear_reg(X_in_cost, y, lambda_test)
    # plot_over_fit_data(X, X_in_cost, y, final_theta)
    # learning_curve(X_in_cost, y, X_val_insert, y_val, lambda_test)
    X_poly = poly_features(X_in_cost, 8)
