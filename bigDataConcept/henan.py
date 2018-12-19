# @Time : 2018/11/3 上午10:39 
# @Author : Kaishun Zhang 
# @File : henan.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
# @Time : 2018/11/2 上午9:05
# @Author : Kaishun Zhang
# @File : cluster_sum.py
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = '/System/Library/Fonts/PingFang.ttc')
def getData():
    with open('henan', 'r') as fw:
        data = fw.read().split('\n')
    X = [];Y = []
    for j in data:
        # print(j.split('   '))
        X.append(j.split('\t')[0])
        Y.append(int(j.split('\t')[1]))
    # print(data_set)
    print(X,Y)
    return (np.array(X),np.array(Y))

def plot_hist():
    plt.figure(figsize = (20, 24))
    # fig.add_subplot(1,2,1)
    X,Y = getData()
    shuffe = np.random.permutation(len(X))
    plt.bar(X[shuffe],Y[shuffe])
    plt.ylim(0,np.max(Y) + 1)
    plt.ylabel("新闻数量",fontproperties = font,fontsize = 20)
    plt.xticks(X[shuffe],fontproperties = font, rotation = 37,fontsize = 16)
    # plt.show()
    plt.savefig('tmp.jpg')
if __name__ == '__main__':
    # getData()
    plot_hist()