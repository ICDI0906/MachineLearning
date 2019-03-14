# @Time : 2018/10/1 下午7:50 
# @Author : Kaishun Zhang 
# @File : viewData.py 
# @Function: 查看一下data 里面的数据

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from MachineLearning.cs229.data.dataAPI import getData
import numpy as np

def plot3D():
    X = pd.read_table('x.dat',header = None, sep='\s+'); Y = pd.read_table('y.dat',header = None, sep='\s+')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    X,Y,Z = X[0],X[1],Y
    # print(X,Y,Z)
    ax = Axes3D(fig) # 画出一个三维图像
    ax.scatter(X, Y, Z)
    plt.show()

def plot2D():
    X,Y = getData()
    X = X.values;Y = Y.values
    zeroPoints = np.array([X[i].tolist() for i in range(X.shape[0]) if Y[i] == 0])
    onePoints = np.array([X[i].tolist() for i in range(X.shape[0]) if Y[i] == 1])
    print(zeroPoints)
    print(onePoints)
    plt.scatter(zeroPoints[:,0],zeroPoints[:,1], marker = 'x',s = 40,linewidths = 2)
    plt.scatter(onePoints[:,0],onePoints[:,1],marker = '^',linewidths = 2)
    plt.show()


if __name__ == '__main__':
    plot2D()