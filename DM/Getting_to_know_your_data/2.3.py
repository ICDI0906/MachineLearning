# @Time : 2018/10/15 下午8:14 
# @Author : Kaishun Zhang 
# @File : 2.3.py 
# @Function:  approximate median
# 1-5 200 6-15 450 16-20 300 21-50 1500 51-80 700 81-110 44
import numpy as np
data = np.array([200,450,300,1500,700,44])
print('median --- > ', 16 + (np.sum(data) / 2 - np.sum(data[:len(data) // 2 - 1])) / (300 + 1500) * 34)