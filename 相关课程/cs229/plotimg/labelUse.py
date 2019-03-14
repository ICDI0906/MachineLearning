# @Time : 2018/10/6 下午7:09 
# @Author : Kaishun Zhang 
# @File : labelUse.py 
# @Function: 看一下 plt 中的 label的使用方式

import numpy as np
import matplotlib.pyplot as plt

def label_use():
    x = np.array(range(10))
    y = x * 2
    y1 = x * 3
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(x,y,label = '-1')
    ax.scatter(x,y1,label = '+1', s =150, c = 'none', alpha = 0.7 ,linewidths = 1.5,edgecolor = '#AB3319')
    plt.show()


if __name__ =='__main__':
    label_use()