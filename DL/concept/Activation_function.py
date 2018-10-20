# @Time : 2018/10/20 下午1:33 
# @Author : Kaishun Zhang 
# @File : Activation_function.py 
# @Function:常见的一些激励函数

import matplotlib.pyplot as plt
import numpy as np

#sigmod
#1/(1+e^(-z))
def sigmod_fun():
    X = np.array(range(-100,100))
    Y = 1.0 / (1 + np.exp(-X))
    plt.plot(X,Y)
    plt.show()

#softplus
#log(1 + e ^ (x))
def softplus_fun():
    X = np.array(range(-100,100))
    Y = np.log(1 + np.exp(X))
    plt.plot(X,Y)
    plt.show()

# softmax
# e^i / e^i + e^j
def softmax_fun():
    X = np.array([-90,0])
    sum = np.sum(np.exp(X))
    Y = np.exp(X) / sum
    print(Y)
    plt.plot(X,Y)
    plt.show()

#tanh
#e^x - e^-x / (e^x - e^-x )
def tanh_fun():
    X = np.array(range(-10,10))
    Y = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    print(Y)
    plt.plot(X,Y)
    plt.show()


if __name__ == '__main__':
    # sigmod_fun()
    # softplus_fun()
    # softmax_fun()
    tanh_fun()
