# @Time : 2018/10/15 下午8:28 
# @Author : Kaishun Zhang 
# @File : 2.4.py 
# @Function: 可视化展示
import numpy as np
x_data = np.array([23,23,27,27,39,41,47,49,50,52,54,54,56,57,58,58,60,61])
y_data = np.array([9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4,30.2,34.2,32.9,41.2,35.7])
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
# plt.scatter(x_data,y_data)
sm.qqplot(y_data,fit = True,line = '45')
plt.show()