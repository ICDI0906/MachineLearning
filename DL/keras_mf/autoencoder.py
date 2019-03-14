# @Time : 2018/10/11 下午7:02 
# @Author : Kaishun Zhang 
# @File : autoencoder.py 
# @Function: 对mnist 数据集进行自编码

import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense,Activation,Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train  = X_train.astype('float32') / 255. - 0.5
X_test = X_test.astype('float32') / 255. - 0.5
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape(X_test.shape[0],-1)

print(X_train.shape)
print(X_test.shape)

encoding_dim = 2
input_img = Input(shape = (784,))

encoded = Dense(128, activation = 'relu')(input_img)
encoded = Dense(64, activation = 'relu')(encoded)
encoded = Dense(10, activation = 'relu')(encoded)
encoder_ouput = Dense(encoding_dim, activation = 'relu')(encoded)

decoded = Dense(10,activation = 'relu')(encoder_ouput)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

autoencoder = Model(input = input_img,output = decoded)
autoencoder.compile(optimizer = 'adam',loss = 'mse')
encoder = Model(input = input_img,output = encoder_ouput)

autoencoder.fit(X_train,X_train,nb_epoch = 10,shuffle = True,batch_size = 256)
autoencoder.save('autoencoder.h5')
# 可视化
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()