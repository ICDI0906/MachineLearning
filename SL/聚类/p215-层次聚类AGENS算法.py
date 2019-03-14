import numpy as np
import sys
import matplotlib.pyplot as plt
Data = np.array([[0.679, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
        [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.260, 0.370], [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
        [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

# 算法思想
# 初始化所有的数据为Data.shape[0]个簇
# 每次迭代从这些簇中找到距离最小两个簇
# 将这两个簇合并起来
# 当簇的个数和需要的簇的个数相等的时候
# 算法结束


def dist(set1, set2):  # 找两个类中距离的最小值
    ans = []
    for i in set1:
        for j in set2:
            ans.append(np.linalg.norm(Data[i]-Data[j]))
    return max(ans)


# 聚类的个数
def agens(n_cluster):
    C = [set() for i in range(Data.shape[0])]
    M = np.zeros((Data.shape[0],Data.shape[0]))  # 初始化一个M*M的矩阵
    for i in range(Data.shape[0]): # 数据的多少就为簇的多少
        C[i] = set([i])
    for i in range(len(C)):
        M[i][i] = sys.maxsize
        for j in range(i+1,len(C)):
            M[i][j] = dist(C[i], C[j])
            M[j][i] = M[i][j]
    q = len(C)
    while q > n_cluster:
        minValue = sys.maxsize
        indexi = -1;indexj = -1
        for i in range(1, len(C)):
            tmpMin = min(M[i])
            if tmpMin < minValue:
                indexi = i; minValue = tmpMin; indexj = np.argmin(M[i])
        C[indexi] = C[indexi] | C[indexj]
        C.remove(C[indexj])
        M = np.zeros((len(C), len(C)))
        for i in range(len(C)):
            M[i][i] = sys.maxsize
            for j in range(i+1, len(C)):
                M[i][j] = dist(C[i],C[j])
                M[j][i] = M[i][j]
        q -= 1
    markers = [",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X"]
    print(n_cluster)
    for i in range(n_cluster):
        plt.scatter(Data[list(C[i]), 0], Data[list(C[i]), 1], s = 40, marker = markers[i], c = 'b', alpha = 0.5)
    plt.show()


agens(5)