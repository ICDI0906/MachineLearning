# @Time : 2018/10/1 上午9:18 
# @Author : Kaishun Zhang 
# @File : LMS.py 
# @Function: least mean square
import numpy as np
import matplotlib.pyplot as plt
import time
# 学习直线 y = a * x + b 中的 a,b 参数
from numpy.linalg import *
def train():
    # np.random.seed(10) # for the same order
    number_sample = 100000
    X = 10 + 10 * np.random.random(number_sample)  # generate 10 number between [10,20)
    X2 = np.array([X,[1 for i in range(number_sample)]]).transpose()

    Y = 2 * X + 4 + np.random.normal(1)
    # plt.subplot(2,2,1)
    # plt.scatter(X,Y)
    # plt.show()
    # 学习2 和 4的参数
    theta = np.random.random(2)
    lr = 1e-5
    print(theta)
    batch_size = 100
    lossList = []

    for i in range(number_sample//batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        theta += lr * np.sum((Y[start : end] - np.dot(X2[start : end], theta)).repeat(2).reshape(batch_size,2) * X2[start:end])
        predY = np.dot(X2,theta.transpose())

        loss = 1 / 2 * np.sum((predY - Y)**2)
        lossList.append(loss)
        print(theta)
    # just one step to get the theta by theta = (XT*X)-1*XT*Y
    oneStepTheta = np.dot(np.dot(inv(np.dot(X2.transpose(), X2)), X2.transpose()), Y)
    predY = np.dot(X2, oneStepTheta.transpose())
    loss = 1 / 2 * np.sum((predY - Y) ** 2)
    print(loss)
    lossList.append(loss)
    plt.plot(range(number_sample//batch_size + 1),lossList)
    # plt.show()
    plt.savefig('loss.jpg')


if __name__ == '__main__':
    train()

# theta 2.16 2.19