# @Time : 2019/4/24 10:02 PM 
# @Author : Kaishun Zhang 
# @File : GAN-base.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from database import *
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Reshape,Conv2D,Activation,Input,UpSampling2D,ZeroPadding2D,Flatten,Lambda
from keras.optimizers import Adam
from keras import backend as K
# img = im.imread(path + '/1.jpg')
#
# print(img)
# print(img.shape)
# plt.imshow(img)
# plt.show()


def loss_func(y_true, y_predict):
    return - K.mean(K.maximum(K.epsilon(), y_predict))


class GANModel():

    def __init__(self):
        self.input = [Input(shape = (10,),name = 'gen'), Input(shape = (96, 96, 3),name = 'dis')]
        ## generator layers
        self.gen =[
            Dense(128 * 24 * 24, activation = 'relu'),
            Reshape((24, 24, 128)),
            UpSampling2D(),
            Conv2D(
                filters = 128,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('relu'),
            UpSampling2D(),
            Conv2D(
                filters = 64,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('relu'),

            Conv2D(
                filters = 3,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('tanh')
        ]
        ## discriminator layer
        self.dis = [
            Reshape((96, 96, 3)),
            Conv2D(
                filters = 32,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('relu'),
            Conv2D(
                filters = 64,
                kernel_size = 4,
                padding = 'same'
            ),
            ZeroPadding2D(),
            Activation('relu'),
            Conv2D(
                filters = 128,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('relu'),
            Conv2D(
                filters = 256,
                kernel_size = 4,
                padding = 'same'
            ),
            Activation('relu'),
            Flatten(),
            Dense(1, activation = 'sigmoid')
        ]
        self.gen_out = self.input[0]
        for l in self.gen:
            self.gen_out = l(self.gen_out)
        self.dis_out = self.input[1]

        # print(self.dis_out)

        for l in self.dis:
            self.dis_out = l(self.dis_out)
        self.optimizer = Adam(lr = 0.0001)

    def get_dis_model(self):
        for l in self.gen:
            l.trainable = False
        for l in self.dis:
            l.trainable = True
        discriminateor_out1 = self.dis_out ##这是真实样本的的输出
        output = self.get_dis()
        discriminateor_out2 = Lambda(lambda y: 1. - y)(output) #这是假样本的输出

        self.__discriminateor_model = Model(self.input, [discriminateor_out2,discriminateor_out1])
        self.__discriminateor_model.compile(self.optimizer,loss = loss_func)

        print(self.__discriminateor_model.summary())

        self.discriminateor_model = Model(self.input[1], self.dis_out)
        self.discriminateor_model.compile(self.optimizer,loss = loss_func)

    def get_gen_model(self):
        for l in self.gen:
            l.trainable = True
        for l in self.dis:
            l.trainable = False

        out = self.get_dis()

        self.__generator_model = Model(self.input[0], out)
        self.__generator_model.compile(self.optimizer,
                                       loss =loss_func)
        self.generator_model = Model(self.input[0], self.gen_out)

    def train(self, sample_list, epoch = 3, batch_size = 32, step_per = 10, plot = False):
        '''
        step_per: 每隔几步训练一次generator
        '''
        sample_noise, sample_true = sample_list["unreal"], sample_list["real"]
        sample_count = sample_noise.shape[0]
        batch_count = sample_count // batch_size
        psudo_y = np.ones((batch_size,), dtype = 'float32')

        for ei in range(epoch):
            print('iterator : ' ,ei)
            for i in range(step_per):
                idx = np.random.randint(0, batch_count - 1)
                batch_noise = sample_noise[idx * batch_size: (idx + 1) * batch_size]
                idx = np.random.randint(0, batch_count - 1)
                batch_sample = sample_true[idx * batch_size: (idx + 1) * batch_size]

                self.__discriminateor_model.train_on_batch({
                    "dis": batch_sample,
                    "gen": batch_noise},
                    [psudo_y, psudo_y])

            idx =  np.random.randint(0, batch_count - 1)
            batch_noise = sample_noise[idx * batch_size: (idx + 1) * batch_size]
            self.__generator_model.train_on_batch(batch_noise, psudo_y)

        test = np.random.random(size = (1,10))
        img = self.generator_model.predict(test)
        plt.imshow(img[0])
        plt.show()

    def get_dis(self):
        result = self.gen_out
        for l in self.dis:
            result = l(result)
        return result
# x = np.random.normal(size = (1,100))
iterator_num = 5
batch_size = 32
db_data = db().get_data()
noise = np.random.random(size = (len(db_data),10))
sample = {
    'real':db_data,
    'unreal': noise
}
gan = GANModel()
gan.get_dis_model()
gan.get_gen_model()

gan.train(sample_list = sample,epoch = iterator_num)