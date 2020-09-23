import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import scipy.optimize as op

# Logistic Regression


# file name
csv_file = '../files/ex2data1.txt'


# read file
def read_data_to_draw(filename):
    save_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        columns = ['ex1', 'ex2', 'value']
        for row in reader:
            save_list.append([Decimal(row[0]), Decimal(row[1]), int(row[2])])
    df = pd.DataFrame(save_list, columns=columns)
    return df


# draw plot
def draw_plot(name):
    # fetch data
    fetch_data = read_data_to_draw(name)

    # choose 0 or 1
    data_when_zero = fetch_data[fetch_data['value'] == 0]
    data_when_one = fetch_data[fetch_data['value'] == 1]

    # draw plot
    plt.plot(data_when_zero['ex1'], data_when_zero['ex2'], 'ro')
    plt.plot(data_when_one['ex1'], data_when_one['ex2'], 'bx')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


# read file save to list
def read_data_to_manage(filename):
    eg_list = []
    value_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            eg_list.append([float(row[0]), float(row[1])])
            value_list.append([int(row[2])])
    return eg_list, value_list


# list to matrix
def list_to_matrix(list_name):
    matrix = np.mat(list_name)
    return matrix
    # print(matrix)


# manage x
def manage_x(x_name):
    df = pd.DataFrame(x_name, columns=['x1', 'x2'])
    df.insert(0, 'add', 1)
    new_x = [[df.iloc[i]['add'], df.iloc[i]['x1'], df.iloc[i]['x2']] for i in range(0, len(df))]
    return new_x
    print(new_x)


# Compute the sigmoid
def sigmoid(z):
    g = 1. / (1 + np.exp(-z))
    return g


# costFunction
def costFunction(theta, x_in, y_in):
    m = len(y_in)
    g = sigmoid(x_in * theta.reshape(-1, 1))
    J = -(y_in.T * np.log(g) + (1 - y_in).T * np.log(1 - g)) / m
    return J


def gradient(theta, x_in, y_in):
    m = len(y_in)
    g = sigmoid(x_in * theta.reshape(-1, 1))
    grad = x_in.T * (g - y_in) / m

    return grad


def plotDecisionBoundary(name, final_theta):
    # fetch data
    fetch_data = read_data_to_draw(name)

    plot_x = np.mat([float(fetch_data['ex1'].min()), float(fetch_data['ex1'].max())]).T
    plot_y = (-1 / final_theta[2]) * (final_theta[1] * plot_x + final_theta[0])
    data_when_zero = fetch_data[fetch_data['value'] == 0]
    data_when_one = fetch_data[fetch_data['value'] == 1]
    # draw plot
    plt.plot(data_when_zero['ex1'], data_when_zero['ex2'], 'ro')
    plt.plot(data_when_one['ex1'], data_when_one['ex2'], 'bx')
    plt.plot(plot_x, plot_y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Decision Boundary')
    plt.show()


# predict function
def predict(theta, XX, yy):
    probability = sigmoid(XX * theta)
    predictions = [1 if x >= 0.5 else 0 for x in probability]  # return a list
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, yy)]
    accuracy = sum(correct) / len(XX)
    print(str(accuracy * 100)[:2] + '%')


if __name__ == '__main__':
    # set theta
    initial_theta = np.zeros([1, 3])

    # test theta
    # test_theta = np.mat([[-24], [0.2], [0.2]])

    # read file
    ex_1_and_2, value = read_data_to_manage(csv_file)
    # manage x
    ex_new = manage_x(ex_1_and_2)
    # to matrix
    X = list_to_matrix(ex_new)
    y = list_to_matrix(value)
    # compute cost and grad
    Cost = costFunction(initial_theta, X, y)
    Grad = gradient(initial_theta, X, y)

    # obtain the optimal theta
    result = op.minimize(fun=costFunction, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
    print(result)
    final_the = result['x']

    # plot decision boundary
    # plotDecisionBoundary(csv_file, final_the)

    # Predict probability for a student with score 45 on exam 1 and score 85 on exam2
    prob = sigmoid(np.mat([1, 45, 85]) * np.mat(final_the).T)
    # print(prob)

    # calculate accuracy
    predict(np.mat(final_the).T, X, y)
