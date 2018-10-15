# @Time : 2018/10/15 上午10:13 
# @Author : Kaishun Zhang 
# @File : boxplot.py 
# @Function: 箱形图画查看数据离群点

import matplotlib.pyplot as plt
import numpy as np


def box_plot():
    data = np.array([47,30,36,50,52,52,56,60,70,70,110])
    a = plt.boxplot(data)
    print(a)
    plt.show()

if __name__ == '__main__':
    box_plot()