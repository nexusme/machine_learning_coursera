import csv
import scipy.optimize as op

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

# file name
csv_file = '../files/ex2data2.txt'


# read file
def read_data(filename):
    save_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        columns = ['t1', 't2', 'value']
        for row in reader:
            save_list.append([float(row[0]), float(row[1]), int(row[2])])
    df = pd.DataFrame(save_list, columns=columns)
    return df


# draw plot
def draw_plot(filename):
    # fetch data
    fetch_data = read_data(filename)

    # choose 0 or 1
    data_when_zero = fetch_data[fetch_data['value'] == 0]
    data_when_one = fetch_data[fetch_data['value'] == 1]

    # draw plot
    plt.plot(data_when_zero['t1'], data_when_zero['t2'], 'ro')
    plt.plot(data_when_one['t1'], data_when_one['t2'], '+')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')


# list to matrix
def list_to_matrix(list_name):
    matrix = np.mat(list_name)
    return matrix
    # print(matrix)


# map(add) feature
def mapFeature(x1, x2):
    power = 6
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)

    return pd.DataFrame(data)


# Compute the sigmoid
def sigmoid(z):
    g = 1. / (1 + np.exp(-z))
    return g


def gradient(theta_in, XX, yy):
    lambda_in = 1
    m = len(yy)
    theta_in = theta_in.reshape(-1, 1)
    hypo = sigmoid(XX @ theta_in)
    grad = ((1 / m) * (hypo - yy).T @ XX).T
    grad[1: len(grad)] = grad[1:len(grad)] + lambda_in / m * theta_in[1:len(theta_in)]
    return grad


# compute cost and grad
def costFunctionReg(theta_in, XX, yy):
    lambda_in = 1
    m = len(yy)
    hypo = sigmoid(XX @ theta_in)
    # print(sum(np.power(theta_in[1:len(theta_in)], 2)))
    J = 1 / m * sum(-yy.T @ np.log(hypo) - (1 - yy).T @ (np.log(1 - hypo))) + (lambda_in / (2 * m) * sum(
        np.power(theta_in[1:len(theta_in)], 2)))

    return J


# decision boundary
def decision_boundary(final_theta_in):
    x = np.linspace(-1, 1.5, 250)
    xx, yy = np.meshgrid(x, x)

    z = mapFeature(xx.ravel(), yy.ravel()).values
    z = z @ final_theta_in
    z = z.reshape(xx.shape)

    draw_plot(csv_file)
    plt.contour(xx, yy, z, 0)
    plt.ylim(-.8, 1.2)
    plt.show()


# predict
def predict(XX, yy, theta_in):
    p = sigmoid(XX @ theta_in)
    p = p.reshape(-1, 1)
    predictions = [1 if x >= 0.5 else 0 for x in p]  # return a list
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, yy)]
    accuracy = sum(correct) / len(XX)
    print(str(accuracy * 100)[:2] + '%')


if __name__ == '__main__':
    # read data from file
    df_read = read_data(csv_file)
    # draw_plot(csv_file)
    df_map = mapFeature(df_read['t1'].values, df_read['t2'].values)

    # initial_theta
    initial_theta = np.zeros([df_map.shape[1], 1])

    # test theta
    test_theta = np.ones([df_map.shape[1], 1])

    # to matrix
    X = df_map.values
    y_list = [[df_read.iloc[i]['value']] for i in range(0, len(df_read))]
    y = list_to_matrix(y_list)

    # compute cost
    grad_r = gradient(test_theta, X, y)
    # obtain the optimal theta
    result = op.minimize(fun=costFunctionReg, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)

    # decision_boundary(result['x'])
    predict(X, y, result['x'])
