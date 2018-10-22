# @Time : 2018/10/22 上午11:04 
# @Author : Kaishun Zhang 
# @File : 3.3.py 
# @Function: 数据平滑

# method to smooth the data
# bin, regression, outlier analysis

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 主要是进行数据平滑还有异常值的检测


def exercises3_1():
    data = np.array(
        [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
    counter = Counter(data)
    dic_count = {}
    for key,value in counter.items():
        if value not in dic_count.keys():
            dic_count[value] = {}
            dic_count[value]['aver'] = key
            dic_count[value]['list'] = []
            dic_count[value]['list'].append(key)
        else:
            dic_count[value]['list'].append(key)
            dic_count[value]['aver'] = np.sum(dic_count[value]['list'])/len(dic_count[value]['list'])
    print(dic_count)
    loc_i = 1
    fig = plt.figure()
    for key ,value in dic_count.items():
        fig.add_subplot(2,2,loc_i)
        plt.boxplot(value['list'])
        loc_i +=1
    plt.show()


if __name__ == '__main__':
    exercises3_1()