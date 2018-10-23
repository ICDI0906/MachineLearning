# @Time : 2018/10/22 下午3:48 
# @Author : Kaishun Zhang 
# @File : regularization.py 
# @Function: 测试一下正则化是怎么来泛化误差的
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout

def regularization_test():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1,784);X_test = X_test.reshape(-1,784)
    Y_train = np_utils.to_categorical(Y_train,num_classes = 10)
    Y_test = np_utils.to_categorical(Y_test,num_classes = 10)
    model = Sequential([
        Dense(32,input_dim = 784,kernel_regularizer = regularizers.l2(0.0001)),  # this parameter has to change
        Activation('relu'),
        Dense(32,kernel_regularizer = regularizers.l2(0.0001)),
        Activation('relu'),
        Dense(32,kernel_regularizer = regularizers.l2(0.0001)),
        Activation('relu'),
        Dense(32,kernel_regularizer = regularizers.l2(0.0001)),
        Dropout(0.25),
        Activation('relu'),
        Dense(10,kernel_regularizer = regularizers.l2(0.0001)),
        Activation('softmax')
    ])
    print(model.summary())
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    model.fit(X_train,Y_train,epochs = 1000,batch_size = 32,validation_data = (X_test,Y_test),callbacks = [EarlyStopping(monitor='val_err', patience=2)])
    # earlyStopping 当验证集的损失出现增高的两次时停止训练

    # print(X_train.shape)
def train_end(str):
    print(str)

if __name__ == '__main__':
    regularization_test()

# conclusion
# 出现准确率不高的时候，这里出现high basis，这时可以适当地增加网络层数，
# 当准确率高，val_loss ，出现high variance。这时可以增加，正则化参数，
# 参数设置过大，则会退化成线性模型。或者增加dropout层来减少一层中单元的个数
