# @Time : 2018/10/15 下午7:40 
# @Author : Kaishun Zhang 
# @File : 2.2.py 
# @Function: 习题2.2

import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

data = np.array([13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,35,35,35,35,36,40,45,46,52,70])
length = len(data)
# mean
print('mean ---- > ', np.mean(data))
# median
print('median ----- > ',data[length // 2] if length % 2 == 1 else (data[length // 2] + data[(length -1) // 2]) / 2)
# mode method one
counts = np.bincount(data)
print('mode ----- > ',np.argmax(counts))
# mode method two
print(stats.mode(data)[0][0])
# Midrange
print('midrange ----- > ',(np.max(data) + np.min(data)) / 2)
# first quartile
print('first quartile ----- > ',np.percentile(data, 25))
# third quartile
print('third quartile  ----- > ',np.percentile(data,75))
plt.boxplot(data)
plt.title('boxplot')
plt.show()

