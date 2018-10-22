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

# 3.11 plot an equal-width histogram of width 10
def histogram10(data = [], step_num = 10):
    data = np.array(
        [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
    count = Counter(data)
    step = range(np.min(data), np.max(data), step_num)
    step = (list(step))
    if step[len(step) - 1] != np.max(data): # 将最后一个间隔加上去
        step.append(np.max(data))
    x_ticks = []
    value = []
    start = step[0]-1
    for i in range(len(step) - 1):
        str_tick = str(start + 1) + '-' + str(step[i + 1])
        x_ticks.append(str_tick)
        cnt = 0
        for data_tmp in data:
            if data_tmp >= (start + 1) and data_tmp <= step[i + 1]:
                cnt += 1
            if data_tmp > step[i + 1]:
                break
        value.append(cnt)
        start = step[i + 1]
    print(value)
    print(x_ticks)
    plt.bar(x_ticks,value)
    plt.savefig('3.11.jpg')
    plt.show()

# Simple random sample without replacement of size s

# param size
# param mode if mode = 1 srswor otherwise srswr


def srswor_srswr(size = 5,mode = 1):
    data = np.array(
        [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
    have_index = set()
    sample_list = []
    while size:
        index = int(np.random.rand() * len(data))
        if mode == 1:
            if index not in have_index:
                have_index.add(index)
                sample_list.append(data[index])
                size -= 1
        else:
            sample_list.append(data[index])
            size -= 1
    print('index ---- >',have_index)
    print('sample_list ------ > ',sample_list)


if __name__ == '__main__':
    # exercises3_1()
    # histogram10()
    srswor_srswr(20,mode = 1)