# @Time : 2018/10/11 下午6:30 
# @Author : Kaishun Zhang 
# @File : rnn.py 
# @Function: simpleRNN 实现mnist 数据集分类

import numpy as np
from  keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense,Activation,SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28,28) / 255
X_test = X_test.reshape(-1,28,28) / 255
y_train = np_utils.to_categorical(y_train, num_classes = 10)
y_test = np_utils.to_categorical(y_test, num_classes = 10)

INPUT_SIZE = 28
TIME_STEP = 28
CELL_SIZE = 50
OUTPUT_SIZE = 10
BATCH_INDEX = 0
BATCH_SIZE = 50
model = Sequential()
model.add(SimpleRNN(
    batch_input_shape = [None,TIME_STEP,INPUT_SIZE],
    output_dim = CELL_SIZE,
    unroll = False
))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
Iterator = 10000
adam = Adam(lr = 1e-4)
model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
for i in range(Iterator):
    print(i)
    X_batch = X_train[BATCH_INDEX : BATCH_INDEX + BATCH_SIZE,:,:]
    y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE,:]
    model.train_on_batch(X_train,y_train)
    BATCH_INDEX = BATCH_INDEX + BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    if i % 10 == 0:
        loss, acc = model.evaluate(X_test, y_test)
        print('loss is {} and the accuracy is {}'.format(loss, acc))


