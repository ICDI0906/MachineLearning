# @Time : 2018/10/2 上午9:15 
# @Author : Kaishun Zhang 
# @File : lw-lr.py 
# @Function: locally-weighted logistic regression 使用的是牛顿法求解

from MachineLearning.cs229.data.dataAPI import getData # must write the full path
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

# param 训练数据集，lable, 测试数据，tau
# return x 判断是否大于零 if > 0 则是1类，否则为0类
def lwlr(trainX,trainY,x,tau):
    (m , n) = trainX.shape
    theta = np.zeros(n)
    w = np.exp(-np.sum((trainX - x)**2,axis = 1) / (2 * tau**2))
    print('w -- > ', w.shape)
    g = np.ones(n)
    while np.linalg.norm(g) >= 1e-6: # 第二范式，就是一阶导数尽量为0的地方
        print(theta)
        h = 1 / (1 + np.exp(-np.dot(trainX,theta.transpose())))
        print('h ---> ',h.shape)
        g = np.dot(trainX.transpose(), (w * (trainY.transpose() - h)).transpose()) .transpose() - 1e-4 * theta  # 对theta 的一阶偏导
        print('g ---> ',g.shape)
        H = np.dot(np.dot(-trainX.transpose(),np.diag(w * h * (1 - h))),trainX) - 1e-4 * np.eye(n) # 这是对theta的二阶导数
        print('h ---> ',H.shape)
        print('theta.shape', theta.shape)
        print('inv.shape ---> ',inv(H).shape)
        print('g.shape ---> ',g.shape)
        print('dot.shape --- > ',np.dot(inv(H), g.transpose()).shape)
        theta = theta - np.dot(inv(H), g.transpose())
    return np.dot(x,theta) > 0


if __name__ == '__main__':
    X,Y = getData()
    X = X.values; Y = Y.values
    yTMP = []
    for val in Y:
        yTMP.append(val[0])
    Y = np.array(yTMP)
    print(Y.shape)
    # print(Y.transpose().shape)
    # (68, 2) (68, 1)
    taus = [0.01, 0.05,0.1, 0.5,1.0,5.0] # check different tau to see
    fig = plt.figure()
    for index,tau in enumerate(taus):
        one = [] ; zero = []
        for i in range(X.shape[0]):
            result = lwlr(X, Y,X[i,:], tau)
            if result == 1:
                one.append(X[i,:].tolist())
            else:
                zero.append(X[i,:].tolist())
        onePoints = np.array(one);zeroPoints = np.array(zero)
        # print(len(taus),i)
        plt.subplot(2,len(taus)/2,index + 1)
        plt.scatter(zeroPoints[:, 0], zeroPoints[:, 1], marker = 'x', s = 40, linewidths = 2)
        plt.scatter(onePoints[:, 0], onePoints[:, 1], marker = '^', linewidths = 2)
        plt.title("tau = {0}".format(tau))
    # plt.show()
    plt.savefig('different_tau.jpg')
    # it prove that when tau is small,it may overfitting,and when tau grows, it has better fitting.
