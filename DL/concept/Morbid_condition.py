# @Time : 2018/10/14 下午7:59 
# @Author : Kaishun Zhang 
# @File : Morbid_condition.py 
# @Function: 验证一下病态条件，就是输入微小变化，然而值却变换很大的情况
import numpy as np
from numpy.linalg import *
# f = A-1 X


def func():
    A = np.array([[1,2],[3,4]])
    x = np.array([3,4])
    X = np.array([i + 1 for i in range(4 * 10)])
    index = 0
    for index in range(10):
        A = X[index * 4 : (index + 1) * 4].reshape((2,2))
        print(np.dot(inv(A),x))


if __name__ == '__main__':
    func()