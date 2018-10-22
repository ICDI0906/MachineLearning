# @Time : 2018/10/22 下午12:28 
# @Author : Kaishun Zhang 
# @File : 3.7.py 
# @Function: normalization
import numpy as np

data = np.array(
        [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70,100])

def min_max():
    return [(x - np.min(data)) / (np.max(data) - np.min(data)) for x in data]

def z_score():
    return (data - np.mean(data)) / np.std(data)

def decimal_scaling():
    max_ = np.max(data)
    cnt = 0;
    while max_:
        cnt += 1
        max_ = max_ // 10  # does not write Binary
    max_tmp = np.power(10, cnt-1)
    if np.max(data) != max_tmp:
        max_ = np.power(10, cnt)
    else:
        max_ = max_tmp
    return data / max_


if __name__ == '__main__':
    print(min_max())
    print(z_score())
    print(decimal_scaling())