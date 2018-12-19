# @Time : 2018/11/29 上午8:34 
# @Author : Kaishun Zhang 
# @File : viewdata.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('/Users/icdi/Downloads/PMU-UD/4/629.jpg')
print(img[:10])
plt.imshow(img)
plt.show()