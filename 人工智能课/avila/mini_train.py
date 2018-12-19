# @Time : 2018/12/1 下午3:50 
# @Author : Kaishun Zhang 
# @File : mini_train.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

frame = pd.read_csv('avila-tr.csv')
frame[:300].to_csv('mini_train2.csv', index = False)
# data = np.round(frame.values[:20,:-1].astype(float),2)
# frame = pd.DataFrame(data)
# frame.to_csv('mini_train1.csv',index = False,index_label = None)