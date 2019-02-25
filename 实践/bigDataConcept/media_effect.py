# @Time : 2018/11/1 上午10:09 
# @Author : Kaishun Zhang 
# @File : media_effect.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = '/System/Library/Fonts/PingFang.ttc')
def getData():
    with open('media_effect', 'r') as fw:
        data = fw.read().split('\n')
    X = [];Y = []
    for j in data:
        X.append(j.split('\t')[0])
        Y.append(int(j.split('\t')[1]))
    return (np.array(X),np.array(Y))

def plot_hist():
    plt.figure(figsize = (8, 10))
    # fig.add_subplot(1,2,1)
    X,Y = getData()
    shuffe = np.random.permutation(len(X))
    plt.bar(X[shuffe],Y[shuffe])
    plt.ylim(4,np.max(Y) + 1)

    plt.xticks(X[shuffe],fontproperties = font, rotation = 35)

    # plt.show()
    plt.savefig('tmp.jpg')
if __name__ == '__main__':
    plot_hist()