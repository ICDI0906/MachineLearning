# @Time : 2018/11/17 下午7:51
# @Author : Kaishun Zhang
# @File : han_format.py
# @Function: 将数据转化为韩老师程序的形式
import numpy as np
import matplotlib.pyplot as plt

with open('../data/data.txt','r') as fw:
    data = fw.read().split('\n')
dic = dict()
mx = -1
for data_tmp in data:
    split_data = data_tmp.split(';')
    mx = max(mx,len(split_data))
    for split_i in split_data:
        if not split_i in dic.keys():
            dic[split_i] = len(dic)
print('total number of transactions',len(data))
print('max len of one transaction: ',mx)
print('number of items',len(dic))
with open('han_format.txt','w') as fw:
    for data_tmp in data:

        split_data = data_tmp.split(';')
        fw.write(str(len(split_data)))
        for data_i in split_data:
            fw.write(" "+ str(dic[data_i]))
        fw.write('\n')

f_w = open('me_result','w')
print(dic)
with open('../FP-growth/result-fp-growth') as fw:
    data = fw.read().split('\n')
    for data_tmp in data:
        data_split = data_tmp.split(';')
        # print(data_split)
        for data_i in data_split[:-1]:
            try:
                print(dic[data_i])
                f_w.write(str(dic[data_i]) + " ")
            except:
                print(data_tmp)
        # print(data_split[-1])
        f_w.write(':' + str(data_split[-1]) + '\n')
