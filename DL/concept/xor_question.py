# @Time : 2018/10/19 下午3:00 
# @Author : Kaishun Zhang 
# @File : xor_question.py 
# @Function: 使用keras 实现oxr，求解参数
from keras.models import Model,Sequential
from keras.layers import Dense,Activation
import numpy as np
from keras.optimizers import SGD


def xor(X,Y):

    model = Sequential([
        Dense(2, input_dim = 2),
        Activation('relu'),
        Dense(1),
        # Activation('softmax')
    ])
    # sgd = SGD(lr = 1e-5, momentum = 0.5 ,decay = 0.09) # momentum 为动态量，decay是每次学习率下降的多少
                                                       # https://blog.csdn.net/bvl10101111/article/details/72615621
    model.compile(loss = 'mse', optimizer = 'sgd')
    print(model.summary())
    model.fit(X,Y,epochs = 100)
    print('in training ...... ')
    print(model.layers[0].get_weights())
    print(model.layers[1].get_weights())


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    xor(X,Y)