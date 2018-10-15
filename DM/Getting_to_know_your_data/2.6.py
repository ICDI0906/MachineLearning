# @Time : 2018/10/15 下午8:42 
# @Author : Kaishun Zhang 
# @File : 2.6.py 
# @Function: 常用的distance
import numpy as np
tup1 = np.array([22,1,42,10])
tup2 = np.array([20,0,36,8])

# euclidean distance
print('euclidean distance --- > ',np.sqrt(np.sum((tup1 - tup2) ** 2)))
# euclidean distance
print('manhattan distance ---- > ', np.sum(np.abs(tup1 - tup2)))
# Minkowski distance
print('Minkowski distance ---- > ',np.power(np.sum((tup1 - tup2)**3),1/3))
# supermum distance
print('supermum distance ---- > ',max(abs(tup1 - tup2)))
