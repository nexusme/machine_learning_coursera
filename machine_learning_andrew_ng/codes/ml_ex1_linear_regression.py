import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

# linear regression


# file name
csv_file = '../files/ex1data1.txt'
# set ite and alpha
iterations = 1500
alpha = 0.01
# set initial theta
theta_test_1 = np.zeros([2, 1])
theta_test_2 = np.mat('-1;2')


# read csv file and save, para:file name, return x, y
def read_data(filename):
    xx = []
    yy = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            xx.append(float(row[0]))
            yy.append(float(row[1]))
    # print(xx)
    # print(yy)
    return xx, yy


# Draw the plot, para x and y, show plot
def draw_plot(data_x, data_y):
    plt.plot(data_x, data_y, 'rx', 10)
    # Set the y−axis label
    plt.ylabel('Profit in $10,000s')
    # Set the x−axis label
    plt.xlabel('Population of City in 10,000s')
    # show plot
    plt.show()


# data to matrix
def data_to_matrix(data_x, data_y):
    # Add a column of ones to x and let x be a matrix
    data_x = [[1, row] for row in data_x]
    matrix_x = np.mat(data_x)
    # let y be a matrix
    data_y = [[row] for row in data_y]
    matrix_y = np.mat(data_y)
    return matrix_x, matrix_y


# compute cost
def compute_cost(data_x, data_y, theta_in):
    # compute sqrErrors
    sqrErrors = np.power((data_x * theta_in - data_y), 2).sum()
    # compute cost
    J = sqrErrors / (2 * len(data_y))
    # print(J)
    return J


# gradient descent
def gradientDescent(data_x, data_y, theta, select_alpha, nums_ite):
    m = len(data_y)
    J_history = np.zeros([nums_ite, 1])
    for ite in range(1, nums_ite):
        theta = theta - select_alpha / m * data_x.T * (data_x * theta - data_y)
        J_history[ite] = compute_cost(data_x, data_y, theta)
    # print(theta, J_history)
    return theta, J_history


# Plot the linear fit
def draw_linear_fit(data_x, matrix_X, data_y, theta_in):
    plt.plot(data_x, data_y, 'x')
    # Set the y−axis label
    plt.ylabel('Profit in $10,000s')
    # Set the x−axis label
    plt.xlabel('Population of City in 10,000s')
    # show plot
    plt.plot(data_x, matrix_X * theta_in, '-')
    plt.show()


# get vals
def get_val(xx, yy):
    #  Grid over which we will calculate J
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)

    # initialize J_val to a matrix of 0's
    J = np.zeros([len(theta0), len(theta1)])

    # Fill out J_val
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            t = np.mat([theta0[i], theta1[j]]).T
            J[i, j] = compute_cost(xx, yy, t)

    J = J.T
    return theta0, theta1, J


# draw Surface plot
def sur_plot(theta0, theta1, J):
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)
    delta = 0.125
    ax.plot_surface(theta0, theta1, J_val, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()


# draw Contour plot
def contour_plot(theta0, theta1, J, X, Y):
    plt.contour(theta0, theta1, J, np.logspace(-2, 3, 20))
    plt.plot(X, Y, 'rx')
    plt.show()


# get x and y
x, y = read_data(csv_file)

# get matrix
matrix_x, matrix_y = data_to_matrix(x, y)

# get theta and cost
final_theta, cost = gradientDescent(matrix_x, matrix_y, theta_test_1, alpha, iterations)

# compute_cost(matrix_x, matrix_y, final_theta)

# draw linear_fit
# draw_linear_fit(x, matrix_x, y, final_theta)

theta0_val, theta1_val, J_val = get_val(matrix_x, matrix_y)

# Surface plot
sur_plot(theta0_val, theta1_val, J_val)

# Contour plot
contour_plot(theta0_val, theta1_val, J_val, final_theta[0][0], final_theta[1][0])
