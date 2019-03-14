# @Time : 2018/10/10 下午1:14 
# @Author : Kaishun Zhang 
# @File : classification.py 
# @Function: 用keras 实现对mnist数据集的分类

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape,X_test.shape)
X_train = np.reshape(X_train,(X_train.shape[0],-1))/255 # 数据归一化之后，最优解的寻优过程会更加地平缓，更容易正确地收敛到最优解
X_test = np.reshape(X_test,(X_test.shape[0],-1))/255
# print(X_train.shape,X_test.shape)
y_train = np_utils.to_categorical(y_train,num_classes = 10)
y_test = np_utils.to_categorical(y_test,num_classes = 10)

model = Sequential([
    Dense(32,input_dim = 784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
    ])

rmsprop = RMSprop(lr = 0.01,rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = rmsprop,loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 100,batch_size = 32)

loss,accuracy = model.evaluate(X_test,y_test)
print('loss is {} accuracy is {}'.format(loss,accuracy))
enumerate