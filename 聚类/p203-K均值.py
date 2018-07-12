# K均值聚类
import numpy as np
import matplotlib.pyplot as plt
Data = np.array([[0.679, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
        [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.260, 0.370], [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
        [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

"""

k 均值聚类的思想：
随机挑选k个向量当做簇的中心
然后，计算所有向量到这些中心的欧几里得距离，
取最小距离对应的下标作为该向量的类别，
以此迭代，当距离不在变化时，算法结束

"""
K = 3 # 簇的个数
U = Data[np.random.choice(Data.shape[0], 3, replace = False)] # 随机选取的k个向量

while(True):
    print("hello world")
    C = [[] for i in range(K)] # 存放所属的类别
    for index, i in enumerate(Data):
        dis = []
        for j in U:
            dis.append(np.linalg.norm(i-j))
        C[np.argmin(dis)].append(index)
    flag = 0
    for c in C:
        print(c)
    for i in range(K):
        u = sum(Data[C[i], :])/len(C[i])
        if not np.array_equal(u, U[i]):
            U[i] = u
        else:
            flag += 1
    if flag == K:
        break
markers = ['^', 'x', 'o', '*', '+']
for i in range(K):
    plt.scatter(Data[C[i], 0], Data[C[i], 1], s = 40, marker = markers[i], c = 'b', alpha = 0.5)
plt.show()