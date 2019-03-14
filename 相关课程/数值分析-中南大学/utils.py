# @Time : 2019/3/10 10:05 AM 
# @Author : Kaishun Zhang 
# @File : utils.py 
# @Function:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def combination(x,y):
    '''
    :param x: vector x
    :param y: vector y
    :return: (x y)
    '''
    return np.vstack((x,y)).T

# 计算的函数
def fun(x):
    return 1 / (1 + x**2)

# Lagrange interpolation 拉格朗日多项式插值
def lagInter(arr,x):
    '''
    @:param arr 为已知点
    @:param x 需要插值的点
    @:return 返回(x,y) np.ndarray()
    '''
    y = []
    n, m = arr.shape
    for x_i in x:  # 需要进行插值的x
        ans = 0
        for i in range(n):
            # yi
            fen_zi = 1
            fen_mu = 1
            for j in range(n):
                if i != j:
                    fen_mu *= (arr[i, 0] - arr[j, 0])
                    fen_zi *= x_i - arr[j, 0]
            ans += fen_zi / fen_mu * arr[i, 1]
        y.append(ans)
    return combination(x,y)


# Aitken 插值方法,使用滚动数组实现
def aitken(arr,x):
    '''
    :param arr:为已知点
    :param x:需要插值的点
    :return: 返回(x, y)
    '''
    x_o = arr[:,0]
    n, m = arr.shape
    y = []
    for x_i in x:
        pre_I = arr[:,1] # 前一个I 记得要重新赋值
        for col_i in range(n - 1):
            curr_I = np.zeros(n)
            for i in np.arange(col_i + 1, n):
                curr_I[i] = pre_I[i] + (pre_I[col_i] - pre_I[i]) / (x_o[col_i] - x_o[i]) * (x_i - x_o[i])
            pre_I = curr_I
        y.append(pre_I[n - 1])
    return combination(x,y)


def neville(arr,x):
    '''
    :param arr:
    :param x:
    :return: 同上
    '''
    x_o = arr[:, 0]
    n, m = arr.shape
    y = []
    for x_i in x:
        pre_I = arr[:, 1]  # 前一个I 记得要重新赋值
        for col_i in range(n - 1):
            curr_I = np.zeros(n)
            for i in np.arange(col_i + 1, n):
                curr_I[i] = pre_I[i] + (pre_I[i - 1] - pre_I[i]) / (x_o[i - col_i - 1] - x_o[i]) * (x_i - x_o[i])
            pre_I = curr_I
        y.append(pre_I[n - 1])
    return combination(x, y)

# 牛顿插值法
def newton(arr,x):
    '''
    :param arr:
    :param x:
    :return: 同上
    '''
    x_o = arr[:, 0]
    n, m = arr.shape
    y = []
    for x_i in x:
        pre_I = arr[:, 1]  # 前一个I 记得要重新赋值
        sum = pre_I[0] # F(x0)
        tmp = 1 # (x-x0)(x-x1)(x-2) 累乘
        for col_i in range(n - 1):
            curr_I = np.zeros(n)
            for i in np.arange(col_i + 1, n):
                curr_I[i] = (pre_I[i - 1] - pre_I[i]) / (x_o[i - col_i - 1] - x_o[i])
            tmp *= (x_i - x_o[col_i])
            sum += tmp * curr_I[col_i + 1]
            pre_I = curr_I
        y.append(sum)
    return combination(x, y)