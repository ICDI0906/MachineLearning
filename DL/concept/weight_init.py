# @Time : 2018/10/23 下午1:41 
# @Author : Kaishun Zhang 
# @File : weight_init.py 
# @Function: 神经网络中权重初始化的方法
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Model, Sequential

# if you use tanh activation w[l]  =  np.random.randn(hidden_weight.shape) * np.sqrt(1.0 / number of unit in layer[l-1])
# if you use relu activation w[l]  =  np.random.randn(hidden_weight.shape) * np.sqrt(2.0 / number of unit in layer[l-1])

# gradient checking
# (f(x0+a) - f(x0-a)) / 2*a 是否约等于 f(x0) 的导数
# attention when use gradient checking
# 1) does not use in training only in checking
# 2) remember regularization
# 3) does not work with dropout