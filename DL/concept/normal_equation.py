# @Time : 2018/10/19 下午1:34 
# @Author : Kaishun Zhang 
# @File : normal_equation.py 
# @Function: 正规化方程求解,其实就是最小二乘法求解

from numpy.linalg import *
import numpy as np

# 计算(XTX)-1 * XT * Y
# param X
def normal_equation(X,Y):
    theta = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),Y)
    print('%.2f %.2f %.2f' % (theta[0],theta[1],theta[2]))


if __name__ == '__main__':
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    Y = np.array([0,1,1,0])
    normal_equation(X,Y)