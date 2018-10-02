# @Time : 2018/10/1 下午7:50 
# @Author : Kaishun Zhang 
# @File : viewData.py 
# @Function: 查看一下data 里面的数据

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
X = pd.read_table('x.dat',header = None, sep='\s+'); Y = pd.read_table('y.dat',header = None, sep='\s+')
fig = plt.figure(figsize=plt.figaspect(0.5))
X,Y,Z = X[0],X[1],Y
# print(X,Y,Z)
ax = Axes3D(fig) # 画出一个三维图像
ax.scatter(X, Y, Z)
plt.show()