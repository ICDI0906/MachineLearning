# @Time : 2018/10/10 上午9:29 
# @Author : Kaishun Zhang 
# @File : lr.py 
# @Function: 用keras Sequential 实现逻辑回归

import numpy as np
np.random.seed(0)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

number = 200
x = np.linspace(-1,1,number)
np.random.shuffle(x)
y = 0.5 * x + 2 + np.random.normal(0,0.05,(number)) # 这个0和0.05 样本的均值和方差，当选择不当时，
                                                    # 会出现梯度爆炸或者loss 上下浮动的现象

y_T = 0.5 * x + 2
square = 1/number * np.sqrt(np.sum((y-y_T)**2))
print(square)

# plt.scatter(x,y)
# plt.show()

X_train,Y_train = x[:150],y[:150]
X_test,Y_test = x[150:],y[150:]
model = Sequential()
model.add(Dense(output_dim = 1,input_dim = 1))
model.compile(loss = 'mae',optimizer = 'sgd') # 使用最小均方误差来作为损失函数，
# 然后使用梯度下降优化模型
# mse mean square error 均方误差是指参数估计值与参数真值之差平方的期望值;
# rmse 上面数值的平方根
# mae mean absolute error 平均绝对误差 是实际数据值和预测值之间的绝对误差。
# rmae 上诉数值的平方根

iterator_num = 300
lossList = []
for i in range(iterator_num):
    loss = model.train_on_batch(X_train,Y_train)
    lossList.append(loss)
    if i % 10 == 0:
        print('after {} step ,the loss is {}'.format(i,loss))
# 查看网络的网络结构
print(model.summary())
print(model.layers[0].get_weights())
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()