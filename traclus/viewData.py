# @Time : 2018/12/14 下午4:30 
# @Author : Kaishun Zhang 
# @File : viewData.py 
# @Function:
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

data_path = 'data_out'
data_list =[]
with open(data_path,'r') as wf:
    data = wf.read().split('\n')
    for data_i in data:
        data_i_list = data_i.rstrip(" ").split(' ')
        data_list.append(data_i_list)
    data_list = np.array(data_list)
    set_cluster = set(data_list[:,-1])
    print(len(set_cluster))
    print(Counter(data_list[:,-1]))
    color = [randomcolor() for i in range(len(set_cluster))]
    for i,cluster in enumerate(set_cluster):
        data_zero = data_list[data_list[:,-1]==cluster]
        for data_zero_i in data_zero:
            plt.plot([float(data_zero_i[0]),float(data_zero_i[2])],[float(data_zero_i[1]),float(data_zero_i[3])],color = color[i])
    # plt.show()
    plt.savefig('hurrcane.jpg')