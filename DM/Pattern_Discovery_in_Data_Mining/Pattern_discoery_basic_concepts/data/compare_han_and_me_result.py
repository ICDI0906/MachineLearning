# @Time : 2018/11/17 下午8:53 
# @Author : Kaishun Zhang 
# @File : compare_han_and_me_result.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
dic = dict()
with open('han_result','r') as fw:
    result_han = fw.read().split('\n')
    for result in result_han:
        split = result.split(':')
        dic[split[0].strip()] = split[1].strip()

with open('me_result','r') as fw:
    result_me = fw.read().split('\n')
    for result in result_han:
        split = result.split(':')
        if dic[split[0].strip()] != split[1].strip():
            print('there is something wrong!!!')