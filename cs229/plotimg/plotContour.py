# @Time : 2018/10/11 上午10:21 
# @Author : Kaishun Zhang 
# @File : plotContour.py 
# @Function: 查看一下高斯分布的等高线
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
from numpy.linalg import *

# mean 均值 协方差矩阵
# cov  协方差矩阵
# size number of size


def generate(mean = [0,0] ,cov = [[1, 0], [0, 1]],size = 1):
    data = []
    for i in range(size):
        x, y = np.random.multivariate_normal(mean, cov)
        data.append([x, y])
    data = np.array(data)
    return data


# 生成三维高斯的等高线
def plotContour():
    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)
    x,y = np.meshgrid(x,y)
    x_mean = 0;y_mean = 0
    sigma = 1
    # z = np.exp(-1/2 * np.dot(np.dot((data - np.array(mean)).transpose(),inv(cov)), data - np.array(mean)))
    # 这个概率密度函数应该怎么来使用呢
    # z = z/(np.sqrt(2 * np.pi) * np.sqrt(det(cov)))

    z = np.exp(-((y - y_mean) ** 2 + (x - x_mean) ** 2) / (2 * (1 ** 2)))
    z = z / (np.sqrt(2 * np.pi) * 1);
    plt.contour(x,y,z)
    # plt.contourf(x,y,z) color for the gap of the contour
    plt.show()


if __name__ == '__main__':
    plotContour()