# @Time : 2018/10/23 下午1:03 
# @Author : Kaishun Zhang 
# @File : Normalize_training_set.py 
# @Function: 对训练数据集进行归一化处理
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Model, Sequential

def data_visualization():
    # x [2,20] y [2,3]
    x1 = np.random.rand(50) * (18) + 2
    x2 = np.random.rand(50) + 2
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.scatter(x1,x2)
    plt.yticks([0,2,4,6,8,10,20])
    plt.title('not normalization')

    # use mean
    data = np.array([x1,x2])
    data = data.T - np.mean(data,axis = 1)
    data_tmp = data
    fig.add_subplot(2,2,2)
    plt.title('mean normalization')
    plt.scatter(data[:,0],data[:,1])
    # use mean 和 方差
    
    data = np.array([x1, x2])
    data = data_tmp / np.mean(data ** 2, axis = 1)
    fig.add_subplot(2, 2, 3)
    plt.title('normalization')
    plt.scatter(data[:, 0], data[:, 1])
    print(np.var(data[:,0]))
    print(np.var(data[:,1]))
    plt.show()


if __name__ == '__main__':
    data_visualization()
