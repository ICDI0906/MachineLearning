# @Time : 2018/10/11 下午4:01
# @Author : Kaishun Zhang
# @File : cnn.py
# @Function: cnn 实现对mnist 图片进行分类

import numpy as np
from keras.layers import Dense,Activation,MaxPooling2D,Flatten,Convolution2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam

(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape,X_test.shape)
X_train = np.reshape(X_train,(X_train.shape[0],-1))/255 # 数据归一化之后，最优解的寻优过程会更加地平缓，更容易正确地收敛到最优解
X_test = np.reshape(X_test,(X_test.shape[0],-1))/255

X_train = X_train.reshape(-1, 1, 28, 28) # n_sample,n_channels,size,size
X_test = X_test.reshape(-1, 1, 28, 28)

# print(X_train.shape,X_test.shape)
y_train = np_utils.to_categorical(y_train,num_classes = 10)
y_test = np_utils.to_categorical(y_test,num_classes = 10)


model = Sequential()

# 卷积 》激励》池化》卷积》池化》全连接》全连接》softmax
model.add(Convolution2D(
    batch_input_shape = (None,1,28,28), # the first parameter should be equal to None so that feed to any number
    filters = 64,      # number of filter
    kernel_size = 2,   # filter size
    strides = 1,       # 步长
    padding = 'same',
    data_format = 'channels_first'
))

model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
    data_format = 'channels_first'
))
# (32,14,14)

model.add(Convolution2D(
    filters = 64,
    kernel_size = 3,
    strides = 1,
    padding = 'same',
    data_format = 'channels_first'
))

model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
    data_format = 'channels_first'
))
# (64,7,7) the second and third is upper_bound
model.add(Flatten()) # to one dim
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)
model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
print(model.summary())
# print(X_train.shape,y_train.shape)
# model.fit(X_train, y_train, epochs = 1, batch_size=32)
#
# loss,accu = model.evaluate(X_test,y_test)
# print('the loss of {} and the accuracy is {}'.format(loss,accu))
