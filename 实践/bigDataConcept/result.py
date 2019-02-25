# @Time : 2018/11/1 下午6:15 
# @Author : Kaishun Zhang 
# @File : result.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
def getData():
    with open('result', 'r') as fw:
        data = fw.read().split('\n')
    data_set = []
    cluster_04 = []
    for j in data:
        if j.split('\t')[1] == '1':
            cluster_04.append((j.split('\t')[0]))
        data_set.append((j.split('\t')[1]))
    counter = Counter(data_set)
    for key,value in counter.items():
        print(key,' ',value)

    with open('cluster_01','w') as fw:
        for cluster in cluster_04:
            fw.write(cluster+'\n')

if __name__ == '__main__':
    getData()