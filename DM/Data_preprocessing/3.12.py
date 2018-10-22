# @Time : 2018/10/22 下午7:48 
# @Author : Kaishun Zhang 
# @File : 3.12.py 
# @Function: 使用iris.data 实现 ChiMerge

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_analysis():
    with open('iris.data','r') as fw:
        data = fw.read().split('\n')
    frame = [data_tmp.split(',') for data_tmp in data]
    columns = ['a','b','c','d','cate']
    frame = pd.DataFrame(frame,columns = columns)
    print(frame['a'].value_counts())
    print(frame['b'].value_counts())
    print(frame['c'].value_counts())
    print(frame['d'].value_counts())
    print(frame['cate'].value_counts())



def chi_merge():
    pass


if __name__ == '__main__':
    data_analysis()