# @Time : 2018/10/27 下午1:31 
# @Author : Kaishun Zhang 
# @File : distribution.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = '/System/Library/Fonts/PingFang.ttc')

dic = {'0-1000':0,'1000-2000':1,'2000-3000':2,'3000-4000':3,
       '4000-5000':4,'5000-6000':5,'6000-7000':6,'7000-8000':7,
       '8000-9000':8,'9000-10000':9,'10000+':10
}
dic_list = ['0-10','10-20','20-30','30-40',
       '40-50','50-60','60-70','70-80',
       '80-90','90-100','100+']

def judge(number):
    if number>=0 and number<1000:
        return '0-1000'
    elif number >= 1000 and number < 2000:
        return '1000-2000'
    elif number >= 2000 and number < 3000:
        return '2000-3000'
    elif number >= 3000 and number < 4000:
        return '3000-4000'
    elif number >= 4000 and number < 5000:
        return '4000-5000'
    elif number >= 5000 and number < 6000:
        return '5000-6000'
    elif number >= 6000 and number < 7000:
        return '6000-7000'
    elif number >= 7000 and number < 8000:
        return '7000-8000'
    elif number >= 8000 and number < 9000:
        return '4000-5000'
    elif number >= 9000 and number < 10000:
        return '4000-5000'
    else:
        return '10000+'

def getData():
    with open('read_numer_distribution.txt', 'r') as fw:
        data = fw.read().split('\n')
    dim = [(int(j.split('\t')[0]), int(j.split('\t')[1])) for j in data]
    dim = np.array(sorted(dim))
    dist = np.zeros(11)
    for d in dim:
        dist[dic[judge(d[0])]] += d[1]
    return dist

def plot_hist():
    plt.figure(figsize = (8, 10))
    # fig.add_subplot(1,2,1)
    plt.bar(dic_list, getData())
    plt.xlabel("阅读数量 / 100" ,fontproperties = font)
    plt.ylabel("新闻数量" ,fontproperties = font)
    # plt.show()
    plt.savefig('tmp.jpg')

def plot_pie():
    data = getData();
    # 调节图形大小，宽，高
    plt.figure(figsize =  (8,10))
    # fig.add_subplot(1,2,2)
    # 定义饼状图的标签，标签是列表
    labels = dic_list
    # 每个标签占多大，会自动去算百分比
    sizes = data / np.sum(data)
    # colors = ['red', 'yellowgreen', 'lightskyblue']
    # 将某部分爆炸出来， 使用括号，将第一块分割出来，数值的大小是分割出来的与其他两块的间隙
    explode = (0.05, 0, 0, 0 , 0 , 0 , 0 , 0 , 0,0,0.05)

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
    plot_hist()
    # plot_pie()
