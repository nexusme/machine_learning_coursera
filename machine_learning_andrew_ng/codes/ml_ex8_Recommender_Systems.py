import csv

import pandas as pd
import scipy
import sklearn
from skimage import io
from sklearn import svm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

from yaml import serialize

path1 = '../files/ex8_movies.mat'

path2 = '../files/ex8_movieParams.mat'

path3 = '../files/movie_ids.txt'


def read_data(path):
    data = loadmat(path)
    data_Y = data['Y']  # num_movies × num_users_matrix
    data_R = data['R']  # R(i, j) = 1 if user j gave a rating to movie i,
    return data_Y, data_R


def read_para(path):
    data = loadmat(path)
    data_X = data['X']
    data_theta = data['Theta']
    data_num_users = data['num_users']  # 用户数943
    data_num_movies = data['num_movies']  # 电影数1682
    data_num_features = data['num_features']  # 特征10

    return data_X, data_theta, data_num_users, data_num_movies, data_num_features


def cofi_cost(params, Y_in, R_in, num_features_in, lambda_in):
    num_movies_in, num_users_in = Y_in.shape
    num_features_in = num_features_in[0][0]
    X_c = params[:num_movies_in * num_features_in].reshape(num_movies_in, num_features_in)  # 5*3
    theta_c = params[num_movies_in * num_features_in:].reshape(num_users_in, num_features_in)  # 4*3

    part1 = np.sum(((X_c @ theta_c.T - Y_in) ** 2) * R_in) / 2
    part2 = lambda_in * np.sum(np.power(theta_c, 2)) / 2
    part3 = lambda_in * np.sum(np.power(X_c, 2)) / 2
    return part1 + part2 + part3


def cofi_gradient(params, Y_in, R_in, num_features_in, lambda_in):
    num_movies_in, num_users_in = Y_in.shape
    # print(num_movies_in, num_users_in)
    X_c = params[:num_movies_in * num_features_in[0][0]].reshape(num_movies_in, num_features_in[0][0])  # 5*3
    theta_c = params[num_movies_in * num_features_in[0][0]:].reshape(num_users_in, num_features_in[0][0])  # 4*3
    inner = (X_c @ theta_c.T - Y_in) * R_in
    # print(inner.shape)
    # print(theta.shape)

    grad_x = inner @ theta_c + lambda_in * X_c
    grad_theta = inner.T @ X_c + lambda_in * theta_c
    # print(np.concatenate((grad_x.flatten(), grad_theta.flatten()), axis=0))
    return np.concatenate((grad_x.flatten(), grad_theta.flatten()), axis=0)


def normalize_rating(Y_in, R_in):
    m, n = Y_in.shape
    Ymean = np.zeros([m, 1])
    Ynorm = np.zeros([m, n])
    for i in range(m):
        idX = [j for j in range(len(R_in[i, :])) if R_in[i, :][j] == 1]
        Ymean[i] = np.mean(Y_in[i, idX])
        Ynorm[i, idX] = Y_in[i, idX] - Ymean[i]
    return Ymean, Ynorm


def read_movies(path):
    f = open(path, "r", encoding="ISO-8859-1")
    movies = []
    for line in f.readlines():
        movies.append(line.split(' ', 1)[-1][:-1])
    return movies
    # df = pd.DataFrame(save_list, columns=columns)
    # return df


if __name__ == '__main__':
    Y, R = read_data(path1)  # Y评分表: 1682*943 R用户是否评分: 1682x943
    X, theta, num_users, num_movies, num_features = read_para(path2)

    # num_users_t = 4
    # num_movies_t = 5
    # num_features_t = 3

    # X = X[0:num_movies_t, 0: num_features_t]  # 5*3 测试集 num_movies  x num_features matrix of movie features
    # theta = theta[0:num_users_t, 0:num_features_t]  # 每个用户的theta4*3
    # Y = Y[0:num_movies_t, 0: num_users_t]  # 评分表5*4
    # R = R[0:num_movies_t, 0: num_users_t]  # 是否评分5*4

    # cofi_cost(np.r_[X.flatten(), theta.flatten()], Y, R, num_users_t, num_movies_t, num_features_t, 1.5)
    # cofi_gradient(np.r_[X.flatten(), theta.flatten()], Y, R, num_users_t, num_movies_t, num_features_t, 1)
    # 先添加一组自定义的用户数据
    my_ratings = np.zeros((1682, 1))
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    Y = np.concatenate((Y, my_ratings), axis=1)
    R = np.concatenate((R, my_ratings > 0), axis=1)
    nu = num_users + 1

    x = np.random.random(size=(num_movies[0][0], num_features[0][0]))
    theta_new = np.random.random(size=(nu[0][0], num_features[0][0]))
    para = np.concatenate((np.random.random((num_movies[0][0], num_features[0][0])),
                           np.random.random((nu[0][0], num_features[0][0]))))

    Y_mean, Y_norm = normalize_rating(Y, R)

    res = minimize(fun=cofi_cost, x0=para, args=(Y_norm, R, num_features, 10), method='TNC',
                   jac=cofi_gradient)

    trained_X = res.x[:num_movies[0][0] * num_features[0][0]].reshape(num_movies[0][0], num_features[0][0])  # 训练好的X

    trained_theta = res.x[num_movies[0][0] * num_features[0][0]:].reshape(nu[0][0], num_features[0][0])  # 训练好的theta

    predict = trained_X @ trained_theta.T  # 预测值

    my_pre = predict[:, -1] + Y_mean.flatten()
    idx = np.argsort(my_pre)[::-1]
    movies_list = read_movies(path3)

    for i in range(10):
        print('Predicting rating: %0.1f, movie: %s.' % (my_pre[idx[i]], movies_list[idx[i]]))
