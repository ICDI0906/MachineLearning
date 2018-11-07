# @Time : 2018/11/7 下午4:50 
# @Author : Kaishun Zhang 
# @File : Apriori.py 
# @Function:频率挖掘Apriori算法
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
min_sup = 771
# 得到数据
def getData():
    with open('data.txt','r') as fw:
        data = fw.read().split('\n')
    return data
    '''
    for dat in data:
        l = len(dat.split(';'))
        s = len(set(dat.split(';')))
        if l != s:
            print('have same entroy')

    data_list = [data_tmp for dat in data for data_tmp in dat.split(';')]
    counter = Counter(data_list)
    fresult1 = open('result1.txt','w')
    for key,value in counter.items():
        if value > 771:
            fresult1.write(str(value) + ':' + key + '\n')
    # print(counter)
    '''

# 将满足min_sup的数据筛选出来


def greater_than_sup(item):
    if item[1] > min_sup:
        return True;
    else:
        return False;


def find_frequent_1_itemset(Data):
    data_list = [data_tmp for dat in Data for data_tmp in dat.split(';')]
    counter = Counter(data_list)
    return counter


def change_to_set(Ltmp):
    result = set()
    for tmp in Ltmp:
        result.add(tmp[0])
    return result

# 查看c 的子集是不是在Ltmp中


def has_infrequent_subset(c,Ltmp):
    result = change_to_set(Ltmp)
    for i in range(len(c)):
        c_tmp = c[:i] + c[i + 1:]
        c_tmp = ';'.join(c_tmp)
        if not c_tmp in result:
            return False
    return True

# 生成下一个


def apriori_gen(Ltmp):
    result = []
    for L_i in range(len(Ltmp)):
        for L_j in range(L_i + 1, len(Ltmp)):
            k = len(Ltmp[L_i][0].split(';'))
            L_i_tmp = Ltmp[L_i][0].split(';')
            L_j_tmp = Ltmp[L_j][0].split(';')
            for i in range(k):
                if i == k - 1 and L_i_tmp[i] < L_j_tmp[i]:
                    c = L_i_tmp + [L_j_tmp[-1]]
                    if has_infrequent_subset(c,Ltmp):
                        result.append(';'.join(c))
                elif i != k - 1 and L_i_tmp[i] != L_j_tmp[i]:
                    break;
    return result

# 判断是否sub_c 的所有元素是否在c中


def judge(sub_c,c):
    sub_c_list = sub_c.split(';')
    c_list = c.split(';')
    for sub_c in sub_c_list:
        if not sub_c in c_list:
            return False
    return True


def apriori(Data):
    L1 = find_frequent_1_itemset(Data)
    L1 = list(filter(greater_than_sup,L1.items()))
    result = dict()
    for l1 in L1:
        result[l1[0]] = l1[1]
    Ltmp = L1
    Ltmp = sorted(Ltmp,key = lambda x:x[0])
    count = 1
    fresult = open('result2.txt','w')
    while len(Ltmp):
        print('迭代的次数: ', count)
        Ck = apriori_gen(Ltmp)
        print(Ck)
        result_tmp = dict()
        for data in Data:
            for ck in Ck:
                if judge(ck,data):
                    # if ck == 'Fast Food;Restaurants':
                    #     fdebug.write(data+'\n')
                    if ck in result_tmp.keys():
                        result_tmp[ck] += 1
                    else:
                        result_tmp[ck] = 1
        Ltmp = list(filter(greater_than_sup, result_tmp.items()))
        # Ltmp = sorted(Ltmp, key = lambda x: x[0])
        for ltmp in Ltmp:
            result[ltmp[0]] = ltmp[1]
        count += 1
    for key, value in result.items():
        fresult.write(str(value)+ ':' + key + '\n')
    fresult.close()


if __name__ == '__main__':
    data = getData()
    apriori(data)
