# @Time : 2018/10/2 上午9:17 
# @Author : Kaishun Zhang 
# @File : dataAPI.py 
# @Function: 返回数据
__all__ = ['getData']
import pandas as pd
import os
# 获取数据
# param 为获取数据的多少
def getData(size = -1):
    pathPre = '/Users/icdi/Desktop/py_ws/MachineLearning/cs229/data/'
    X = pd.read_table(pathPre + 'x.dat',header = None, sep='\s+'); Y = pd.read_table(pathPre + 'y.dat',header = None, sep='\s+') # must write the full path
    # X,Y,Z = X[0],X[1],Y
    return X[:size],Y[:size]