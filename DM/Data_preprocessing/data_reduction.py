# @Time : 2018/10/18 上午9:43 
# @Author : Kaishun Zhang 
# @File : wavelet_transform.py 
# @Function: 数据降维中方法，包含离散小波变换，主成分分析，维度减少
# study website https://www.cnblogs.com/mikewolf2002/p/3429711.html
# PCA优化的是 最大化 单个属性的方差，而最小化属性之间的协方差，而对角矩阵有很好的性质
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pywt.data
from numpy.linalg import *

def wavelet_transform():
    pass

#param k 降到的维度
def simple_pca(k):
    data_pre = np.array([[-1,-2],[-1,0],[0,0],[2,1],[0,1]])
    if k <= 0 or k > data_pre.shape[1]:
        print('dimension invalid')
        return
    print('before pca ----> ',data_pre)
    data = 1 / data_pre.shape[0] * np.dot(data_pre.T,data_pre)
    feature_value, feature_vector = eig(data)  # 特征是按照列来进行排列的
    print(feature_vector[:,0].shape)           # 选取第一个特征向量作为维度较少的基向量
    print('after pca ----- >',np.dot(feature_vector[:,0],data_pre.T))


if __name__ == '__main__':
    # wavelet_transform()
    simple_pca(1)
