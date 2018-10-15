# @Time : 2018/10/12 上午9:37 
# @Author : Kaishun Zhang 
# @File : checkdataCorrelation.py 
# @Function: 验证一下，数据相关性 概率论和数理统计 p24页
import numpy as np
import matplotlib.pyplot as plt
def calc_r():
    # x = np.linspace(-10,10,10)
    # y = 2 * np.linspace(10,20,10)
    x = np.random.randn(10)
    y = np.random.randn(10)

    std_x = np.std(x);std_y = np.std(y)
    mean_x = np.mean(x);mean_y = np.mean(y)
    r = np.sum((x - mean_x) * (y - mean_y)) / (len(x) * std_x * std_y)
    print(r,'正相关' if r > 0 else '负相关')
    plt.scatter(x,y)
    plt.show()

if __name__ == '__main__':
    calc_r()
