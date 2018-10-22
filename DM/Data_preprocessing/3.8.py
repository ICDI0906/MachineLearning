# @Time : 2018/10/22 下午6:08 
# @Author : Kaishun Zhang 
# @File : 3.8.py 
# @Function: 计算两个变量的的相关性
import numpy as np
import matplotlib.pyplot as plt

# cor(x,y) = E(xy) - mean(x)*mean(y)
def attribute_corr():
    x_data = np.array([23,23,27,27,39,41,47,49,50,52,54,54,56,57,58,58,60,61])
    y_data = np.array([9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4,30.2,34.2,32.9,41.2,35.7])
    x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_corr = np.sum(x_data * x_data) / len(x_data) - np.mean(x_data) * np.mean(x_data)
    y_corr = np.sum(y_data * y_data) / len(y_data) - np.mean(y_data) * np.mean(y_data)

    corr = np.sum(x_data * y_data) / len(x_data) - np.mean(x_data) * np.mean(y_data)
    corr_tmp = np.cov(x_data,y_data)
    print('x_data -- > ',x_data,'y_data ---> ',y_data,'corr ---- > ',corr,'corr_tmp ---- > ',corr_tmp,'x_corr -- > ',x_corr,
          'y_corr --- >',y_corr)
    plt.scatter(x_data,y_data)
    plt.show()


if __name__ == '__main__':
    attribute_corr()
