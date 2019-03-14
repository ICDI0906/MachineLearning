# @Time : 2018/10/5 下午9:54 
# @Author : Kaishun Zhang 
# @File : CA.py 
# @Function: Coordinate ascent 坐标上升求解极值
# f(x1,x2) = x1^2 + x2^2 + x1 * x2
import numpy as np
import matplotlib.pyplot as plt


def plotContour():
    x1 = np.linspace(-10,10,50)
    x2 = np.linspace(-10,10,50)
    # print(x1,x2)
    x1, x2 = np.meshgrid(x1, x2)
    y = x1**2 + x2 **2 + x1 * x2
    plt.figure()
    plt.contour(x1,x2,y)
    step = 5
    _x1 = -5 ; _x2 =-5
    x1L = [_x1]
    x2L = [_x2]
    while step:  # for every step update x1,x2 twice
        _x1 = 1 / 2 * x2L[len(x2L)-1]
        x1L.append(_x1)
        x2L.append(x2L[len(x2L)-1])
        _x2 = 1 / 2 * x1L[len(x1L)-1]
        x1L.append(x1L[len(x1L)-1])
        x2L.append(_x2)
        step -=1
    print(x1L,x2L)
    plt.plot(x1L,x2L)
    plt.title("Coordinate Ascent")
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.show()
    plt.savefig('Coordinate-ascent.jpg')


if __name__ =='__main__':
    plotContour()