# @Time : 2018/10/31 下午9:37 
# @Author : Kaishun Zhang 
# @File : daily_number_of_fafang.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
font = FontProperties(fname = '/System/Library/Fonts/PingFang.ttc')

def getData():
    with open('daily_hour_of_fafagng', 'r') as fw:
        data = fw.read().split('\n')
    dim = [int(j.split('\t')[1]) for j in data]
    return np.array(dim)

def plot_plot():
    # plt.grid(True)
    data = getData()
    X = np.array(range(len(data)))
    # get special point
    special_point_x = [];special_point_y=[]
    mx_index = np.argmax(data)
    special_point_x.append(mx_index)
    special_point_y.append(data[mx_index])
    tmp_data = data.copy()
    tmp_data[mx_index] = 0
    se_mx_index = np.argmax(tmp_data)
    # special_point_x.append(se_mx_index)
    # special_point_y.append(data[se_mx_index])

    plt.text(special_point_x[0], special_point_y[0], ('最大值', special_point_y[0]), ha = 'center', va = 'bottom', fontsize = 10, fontproperties = font)
    plt.text(17, data[17], ('次大值', data[17]), ha = 'center', va = 'bottom', fontsize = 10, fontproperties = font)
    special_point_x.append(17)
    special_point_y.append(data[17])
    plt.plot(X,data)
    plt.scatter(special_point_x,special_point_y,marker = 'o')
    line_y = np.linspace(0,special_point_y[0],100).tolist() + np.linspace(0,data[17],100).tolist()
    line_x = [special_point_x[0] for i in range(100)] + [17 for i in range(100)]
    plt.plot(line_x[:100],line_y[:100],linestyle = '--',c = 'black')
    plt.plot(line_x[100:], line_y[100:], linestyle = '--', c = 'black')


    plt.xticks(X)
    plt.savefig('tmp.jpg')


if __name__ == '__main__':
    plot_plot()