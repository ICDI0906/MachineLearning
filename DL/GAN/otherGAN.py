# @Time : 2019/4/25 2:47 PM 
# @Author : Kaishun Zhang 
# @File : otherGAN.py 
# @Function:
import os
from os import path
import argparse
import logging
import traceback
import random
import pickle
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import layers
from keras import callbacks, regularizers, activations
from keras.engine import Model
from keras.utils.vis_utils import plot_model
import keras.backend as K
from collections import defaultdict
from matplotlib import pyplot as plt
# import app_logger

# loger = logging.getLogger(__name__)


# 注意pred不能为负数，因为pred是一个概率。所以最后一个激活函数的选择要注意
def log_loss_discriminator(y_true, y_pred):
    return - K.log(K.maximum(K.epsilon(), y_pred))


def log_loss_generator(y_true, y_pred):
    return K.log(K.maximum(K.epsilon(), 1. - y_pred))


class GANModel:
    def __init__(self,
                 input_dim):
        '''
            __tensor[0]: 定义了discriminateor的表达式
            __tensor[1]: 定义了generator的表达式
        '''
        # discriminateor 对y进行判别，true samples
        # generator 对x进行生成，noise samples
        if isinstance(input_dim, list):
            input_dim_y, input_dim_x = input_dim[0], input_dim[1]
        elif isinstance(input_dim, int):
            input_dim_x = input_dim_y = input_dim
        else:
            raise ValueError("input_dim should be list or interger, got %r" % input_dim)

        self.__inputs = [layers.Input(shape = (input_dim_y,), name = "y"),
                         layers.Input(shape = (input_dim_x,), name = "x")]
        self.__tensors = [None, None]
        self._discriminate_layers = []
        self._generate_layers = []
        self.train_status = defaultdict(list)

    def add_gen_layer(self, layer):
        self._add_layer(layer, True)

    def add_discr_layer(self, layer):
        self._add_layer(layer)

    def _add_layer(self, layer, for_gen = False):
        idx = 0
        if for_gen:
            self._generate_layers.append(layer)
            idx = 1
        else:
            self._discriminate_layers.append(layer)

        if self.__tensors[idx] is None:
            self.__tensors[idx] = layer(self.__inputs[idx])
        else:
            self.__tensors[idx] = layer(self.__tensors[idx])

    def compile_discriminateor_model(self, optimizer = optimizers.Adam()):
        if len(self._discriminate_layers) <= 0:
            raise ValueError("you need to build discriminateor model before compile it")
        if len(self._generate_layers) <= 0:
            raise ValueError("you need to build generator model before compile discriminateo model")

        for l in self._discriminate_layers:
            l.trainable = True # 还可以这么干
        for l in self._generate_layers:
            l.trainable = False
        discriminateor_out1 = self.__tensors[0]
        discriminateor_out2 = layers.Lambda(lambda y: 1. - y)(self._discriminate_generated())
        self.__discriminateor_model = Model(self.__inputs, [discriminateor_out1, discriminateor_out2])
        self.__discriminateor_model.compile(optimizer,
                                            loss = log_loss_discriminator)

        # 这个才是需要的discriminateor model
        self.discriminateor_model = Model(self.__inputs[0], self.__tensors[0])
        print('discriminateor.summary() :',self.discriminateor_model.summary())
        self.discriminateor_model.compile(optimizer,
                                          loss = log_loss_discriminator)
        # if self.log_dir is not None:
        #    plot_model(self.__discriminateor_model, self.log_dir + "/gan_discriminateor_model.png", show_shapes = True)

    def compile_generator_model(self, optimizer = optimizers.Adam()):
        if len(self._discriminate_layers) <= 0:
            raise ValueError("you need to build discriminateor model before compile generator model")
        if len(self._generate_layers) <= 0:
            raise ValueError("you need to build generator model before compile it")

        for l in self._discriminate_layers:
            l.trainable = False
        for l in self._generate_layers:
            l.trainable = True

        out = self._discriminate_generated()

        self.__generator_model = Model(self.__inputs[1], out)
        self.__generator_model.compile(optimizer,
                                       loss = log_loss_generator)
        # 这个才是真正需要的模型
        self.generator_model = Model(self.__inputs[1], self.__tensors[1])
        print('generator.summary: ',self.generator_model.summary())
        # if self.log_dir is not None:
        #    plot_model(self.__generator_model, self.log_dir + "/gan_generator_model.png", show_shapes = True)

    def train(self, sample_list, epoch = 3, batch_size = 32, step_per = 10, plot = False):
        '''
        step_per: 每隔几步训练一次generator
        '''
        sample_noise, sample_true = sample_list["x"], sample_list["y"]
        sample_count = sample_noise.shape[0]
        batch_count = sample_count // batch_size
        psudo_y = np.ones((batch_size,), dtype = 'float32')
        if plot:
            # plot the real data
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.ion()
            plt.show()
        for ei in range(epoch):
            for i in range(step_per):
                idx = random.randint(0, batch_count - 1)
                batch_noise = sample_noise[idx * batch_size: (idx + 1) * batch_size]
                idx = random.randint(0, batch_count - 1)
                batch_sample = sample_true[idx * batch_size: (idx + 1) * batch_size]
                self.__discriminateor_model.train_on_batch({
                    "y": batch_sample,
                    "x": batch_noise},
                    [psudo_y, psudo_y])

            idx = random.randint(0, batch_count - 1)
            batch_noise = sample_noise[idx * batch_size: (idx + 1) * batch_size]
            self.__generator_model.train_on_batch(batch_noise, psudo_y)

            if plot:
                gen_result = self.generator_model.predict_on_batch(batch_noise)
                self.train_status["gen_result"].append(gen_result)
                dis_result = self.discriminateor_model.predict_on_batch(gen_result)
                self.train_status["dis_result"].append(dis_result)
                freq_g, bin_g = np.histogram(gen_result, density = True)
                # norm to sum1
                freq_g = freq_g * (bin_g[1] - bin_g[0])
                bin_g = bin_g[:-1]
                freq_d, bin_d = np.histogram(batch_sample, density = True)
                freq_d = freq_d * (bin_d[1] - bin_d[0])
                bin_d = bin_d[:-1]
                ax.plot(bin_g, freq_g, 'go-', markersize = 4)
                ax.plot(bin_d, freq_d, 'ko-', markersize = 8)
                gen1d = gen_result.flatten()
                dis1d = dis_result.flatten()
                si = np.argsort(gen1d)
                ax.plot(gen1d[si], dis1d[si], 'r--')
                if (ei + 1) % 20 == 0:
                    ax.cla()
                plt.title("epoch = %d" % (ei + 1))
                plt.pause(0.05)
        if plot:
            plt.ioff()
            plt.close()

    # def save_model(self, path_dir):
    #     self.generator_model.save(path_dir + "/gan_generator.h5")
    #     self.discriminateor_model.save(path_dir + "/gan_discriminateor.h5")

    # def load_model(self, path_dir):
    #     from keras.models import load_model
    #     custom_obj = {
    #         "log_loss_discriminateor": log_loss_discriminateor,
    #         "log_loss_generator": log_loss_generator}
    #     self.generator_model = load_model(path_dir + "/gan_generator.h5", custom_obj)
    #     self.discriminateor_model = load_model(path_dir + "/gan_discriminateor.h5", custom_obj)

    def _discriminate_generated(self):
        # 必须每次重新生成一下
        disc_t = self.__tensors[1]
        for l in self._discriminate_layers:
            disc_t = l(disc_t)
        return disc_t


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("""gan model demo (gaussian sample)""")
    # parser.add_argument("-m", "--model_dir")
    # parser.add_argument("-log", "--log_dir")
    # parser.add_argument("-b", "--batch_size", type = int, default = 32)
    # parser.add_argument("-log_lvl", "--log_lvl", default = "info",
    #                     metavar = "可以指定INFO，DEBUG，WARN, ERROR")
    # parser.add_argument("-e", "--epoch", type = int, default = 10)
    #
    # args = parser.parse_args()
    #
    # log_lvl = {"info": logging.INFO,
    #            "debug": logging.DEBUG,
    #            "warn": logging.WARN,
    #            "warning": logging.WARN,
    #            "error": logging.ERROR,
    #            "err": logging.ERROR}[args.log_lvl.lower()]
    # app_logger.init(log_lvl)
    #
    # loger.info("args: %r" % args)
    step_per = 20
    sample_size = 32 * 100
    epoch = 10000
    batch_size = 32
    # 整个测试样本集合
    noise_dim = 4
    signal_dim = 1
    x = np.random.uniform(-3, 3, size = (sample_size, noise_dim))
    y = np.random.normal(size = (sample_size, signal_dim))
    samples = {"x": x,
               "y": y}

    gan = GANModel([signal_dim, noise_dim])
    gan.add_discr_layer(layers.Dense(200, activation = "relu"))
    gan.add_discr_layer(layers.Dense(50, activation = "softmax"))
    gan.add_discr_layer(layers.Lambda(lambda y: K.max(y, axis = -1, keepdims = True),
                                      output_shape = (1,)))

    gan.add_gen_layer(layers.Dense(200, activation = "relu"))
    gan.add_gen_layer(layers.Dense(100, activation = "relu"))
    gan.add_gen_layer(layers.Dense(50, activation = "relu"))
    gan.add_gen_layer(layers.Dense(signal_dim))

    gan.compile_generator_model()
    # loger.info("compile generator finished")
    print("compile generator finished")
    gan.compile_discriminateor_model()
    # loger.info("compile discriminator finished")
    print("compile discriminator finished")

    gan.train(samples, epoch, batch_size, step_per, plot = False)
    gen_results = gan.train_status["gen_result"]
    dis_results = gan.train_status["dis_result"]

    # gen_result = gen_results[-1]
    # dis_result = dis_results[-1]
    # freq_g, bin_g = np.histogram(gen_result, density = True)
    # # norm to sum1
    # freq_g = freq_g * (bin_g[1] - bin_g[0])
    # bin_g = bin_g[:-1]
    # freq_d, bin_d = np.histogram(y, bins = 100, density = True)
    # freq_d = freq_d * (bin_d[1] - bin_d[0])
    # bin_d = bin_d[:-1]
    # plt.plot(bin_g, freq_g, 'go-', markersize = 4)
    # plt.plot(bin_d, freq_d, 'ko-', markersize = 8)
    # gen1d = gen_result.flatten()
    # dis1d = dis_result.flatten()
    # si = np.argsort(gen1d)
    # plt.plot(gen1d[si], dis1d[si], 'r--')
    # plt.savefig("img/gan_results.png")

    # if not path.exists(args.model_dir):
    #     os.mkdir(args.model_dir)
    # gan.save_model(args.model_dir)