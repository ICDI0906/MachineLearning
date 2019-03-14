import numpy as np
import matplotlib.pyplot as plt
import random
import queue
Data = np.array([[0.679, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
        [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.260, 0.370], [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
        [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

# 算法思想
# 首先求得所有的核心对象，以及核心对象中点的个数和点
# 随机选取一个核心对象
# 找该核心对象的密度可达对象，并把这些对象做为一个簇
# 循环，直到没有核心对象为至

# params 分别为距离和集群中最少个数
def dbscan(e, minpts):
    C = []
    kernelObject = set() # 核心对象集合
    cnt = [0 for i in range(Data.shape[0])]             # 以Data[i]为核心对象集合的个数
    otherOject = []      # 以Data[i]为核心的对象集合
    # 求取开始的kernel集合
    for i in range(Data.shape[0]):
        tmpSet = set()
        for j in range(Data.shape[0]):
            if np.linalg.norm(Data[i]-Data[j]) <= e:
                cnt[i] += 1;
                tmpSet.add(j)
        otherOject.append(tmpSet)
        if cnt[i] >= minpts:
            kernelObject.add(i)

    unVisit = set([i for i in range(Data.shape[0])]) # 初始化未访问的集合
    n_cluster = 0 # 初始化聚类的个数
    while len(kernelObject) > 0:
        object = random.sample(kernelObject, 1)       # 从集合中随机选取一个对象
        oldUnVisit = unVisit
        unVisit = unVisit - set(object)
        que = queue.Queue()
        que.put(object[0])
        while not que.empty():
            front = que.get()
            if cnt[front] >= minpts:
                tmp = otherOject[front] & unVisit # 将未访问的并且在以front密度直达的求出
                for item in tmp:
                    que.put(item)
                unVisit = unVisit - tmp
        n_cluster += 1
        C.append(list(oldUnVisit - unVisit))

        print(C[n_cluster-1])
        kernelObject = kernelObject - (oldUnVisit-unVisit)
    markers = [",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X"]
    print(n_cluster)
    for i in range(n_cluster):
        plt.scatter(Data[C[i], 0], Data[C[i], 1], s = 40, marker = markers[i], c = 'b', alpha = 0.5)
    plt.show()

dbscan(0.10, 5)

