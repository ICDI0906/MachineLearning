# @Time : 2018/11/2 上午9:05 
# @Author : Kaishun Zhang 
# @File : cluster_sum.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
def getData():
    with open('cluster_sum', 'r') as fw:
        data = fw.read().split('\n')
    data_set = []
    for j in data:
        # print(j.split('   '))
        data_set.append([int(j.split('   ')[0]),int(j.split('   ')[1])])
    # print(data_set)
    return np.array(data_set)

def plot_pie():
    data = getData();
    # 调节图形大小，宽，高
    plt.figure(figsize =  (8,10))
    # fig.add_subplot(1,2,2)
    # 定义饼状图的标签，标签是列表
    labels = data[:,0]
    # 每个标签占多大，会自动去算百分比
    sizes = data[:,1] / np.sum(data[:,1])
    # colors = ['red', 'yellowgreen', 'lightskyblue']
    # 将某部分爆炸出来， 使用括号，将第一块分割出来，数值的大小是分割出来的与其他两块的间隙
    explode = (0.05, 0.05, 0, 0, 0, 0, 0, 0, 0,0)

    patches, l_text, p_text = plt.pie(sizes, explode = explode, labels = labels,
                                      labeldistance = 1.05, autopct = '%3.1f%%', shadow = False,
                                      startangle = 90, pctdistance = 0.6)

    # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
    # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
    # shadow，饼是否有阴影
    # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
    # pctdistance，百分比的text离圆心的距离
    # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

    # 改变文本的大小
    # 方法是把每一个text遍历。调用set_size方法设置它的属性
    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plt.axis('equal')
    plt.legend()
    # plt.show()
    plt.savefig('tmp.jpg')


if __name__ == '__main__':
    # getData()
    plot_pie()