import numpy as np
import math
from numpy.linalg import *
import matplotlib.pyplot as plt
Data = np.array([[0.679, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
        [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.260, 0.370], [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
        [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

# 求得p(x)
def P(x, u, sigma):
    tmp = np.matmul(np.matmul(np.matrix(x - u), inv(np.matrix(sigma))),np.matrix(x-u).transpose())
    fenzi = math.exp(-1 / 2 * tmp[0][0])
    fenmu = pow(2 * math.pi, sigma.shape[0]/2) * math.sqrt(det(sigma))
    return fenzi / fenmu

# @param sigma 为协方差矩阵
# 效率是 iterator_sum * N(数据样例个数) * c_cluster^2
def gaussianMixtureCluster(alpha, n_cluster, sigma, iteratorSum):
    alphaV = np.array([alpha for i in range(n_cluster)])
    sigmaV = np.array([sigma for i in range(n_cluster)])
    uV = Data[np.random.choice(Data.shape[0], n_cluster, replace = False)] # 随机选择n_cluster 个向量作为均值向量
    for repeat in range(iteratorSum):
        R = np.zeros((Data.shape[0],n_cluster)) # R(i,j)表示样本i,属于j的后验概率
        for i in range(Data.shape[0]):
            for j in range(n_cluster):
                fenzi = alphaV[j] * P(Data[i], uV[j], sigmaV[j])
                fenmu = 0.0
                for k in range(n_cluster):
                    fenmu += alphaV[k] * P(Data[i], uV[k], sigmaV[k])
                R[i][j] = fenzi / fenmu
        # print("here")
        for i in range(n_cluster):
            sum_fenzi = np.zeros(Data[0].shape)
            sum_fenmu = 0.0
            sum_sigma = np.zeros(sigmaV[0].shape)
            for j in range(Data.shape[0]):
                sum_fenzi += R[j][i] * Data[j];sum_fenmu += R[j][i]
            uV[i] = sum_fenzi / sum_fenmu
            for j in range(Data.shape[0]):
                sum_sigma += R[j][i] * np.matmul(np.matrix(Data[j] - uV[i]).transpose(), np.matrix(Data[j] - uV[i]))
            sigmaV[i] = sum_sigma / sum_fenmu

            alphaV[i] = sum_fenmu / Data.shape[0]
    C = [[] for i in range(n_cluster)]
    for i in range(Data.shape[0]):
        C[np.argmax(R[i])].append(i)
    markers = ['^', 'x', 'o', '*', '+']
    for i in range(n_cluster):
        plt.scatter(Data[C[i], 0], Data[C[i], 1], s = 40, marker = markers[i], c = 'b', alpha = 0.5)
    plt.show()

gaussianMixtureCluster(float(1/3), 3, np.array([[0.1, 0.0], [0.0, 0.1]]), int(2000))