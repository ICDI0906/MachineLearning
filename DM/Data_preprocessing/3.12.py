# @Time : 2018/10/22 下午7:48 
# @Author : Kaishun Zhang 
# @File : 3.12.py 
# @Function: 使用iris.data 实现 ChiMerge

import numpy as np
import pandas as pd


def data_analysis():
    with open('iris.data','r') as fw:
        data = fw.read().split('\n')
    frame = [data_tmp.split(',') for data_tmp in data]
    columns = ['a','b','c','d','cate']
    frame = pd.DataFrame(frame,columns = columns)
    return frame

# merge adjacent with the smallest chi square util terminate condition is met
def sub_chi_merge(dict_metric,interval):
    sorted_list = sorted(dict_metric.items(),key = lambda key:key[0])
    print('sorted list is  ',sorted_list)
    size = len(sorted_list)
    while size - interval > 0:
        chi_square_value = []
        for i in range(size - 1):
            chi_square_value.append(chi_square(np.array([sorted_list[i][1],sorted_list[i + 1][1]])))
        min_index = np.argmin(np.array(chi_square_value))
        # add value
        for i in range(len(sorted_list[min_index][1])):
            sorted_list[min_index][1][i] += sorted_list[min_index + 1][1][i]
        sorted_list.pop(min_index + 1)
        size = len(sorted_list)
    print('the interval is ')
    for item in sorted_list:
        print(item[0],'value : ',item[1])


def chi_merge(interval = 6):
    frame = data_analysis()
    frame_cate = frame['cate'].value_counts().keys()
    print('frame_cate include -------- >', frame_cate)
    for col in frame.columns:
        if col == 'cate':
            continue
        result = dict()
        frame_col = frame[col].value_counts()
        for key in frame_col.keys():
            if key == '':
                print('there is something wrong')
            result[key] = []
            value_count_tmp = frame[frame[col] == key]['cate'].value_counts()
            for cate_name in frame_cate:
                if cate_name in value_count_tmp.keys():
                    result[key].append(value_count_tmp[cate_name])
                else:
                    result[key].append(0)
        sub_chi_merge(result,interval)

# check the greater the value ,the difference the distribution
# when the value equals zero ,the distribution are the same
#param data matrix


def chi_square(data = np.array([])):
    # data = np.array([[250,200],[50,1000]])
    sum_y = np.sum(data,axis = 1)
    sum_x = np.sum(data,axis = 0)
    # print(sum_x,sum_y)
    e_ij = np.array([i * j / np.sum(sum_y) for i in sum_y for j in sum_x]).reshape(data.shape)
    ans = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if e_ij[i][j] == 0:     # be careful for this
                continue
            else:
                ans.append((data[i][j] - e_ij[i][j]) ** 2 / e_ij[i][j])
    return np.sum(ans)


if __name__ == '__main__':
    chi_merge()