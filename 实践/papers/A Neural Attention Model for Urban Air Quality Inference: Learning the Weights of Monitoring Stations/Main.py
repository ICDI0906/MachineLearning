# @Time : 2019/6/10 12:54 PM 
# @Author : Kaishun Zhang 
# @File : Main.py 
# @Function:
from utils import *
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell,MultiRNNCell
from tensorflow.contrib.layers import repeat
from sklearn.metrics import mean_squared_error

LAYER = 2
BATCH_SIZE = 1
HIDDEN_SIZE = 300
keep_prob = 0.1
fc1_unit = 100
fc2_unit = 200
batch_size = 64
time_window = 48
logger = get_logger('.', name = __name__)


# random 10
def Main():
    ###
    # build model right
    ###
    tf.reset_default_graph()
    with tf.variable_scope('input'):
        source_input = tf.placeholder(dtype = tf.float32,shape = (None,time_window,576), name = 'source_input')
        target_input_mete = tf.placeholder(dtype = tf.float32,shape = (None,time_window,31), name = 'target_input_mete')
        source_dist_angle = tf.placeholder(dtype = tf.float32,shape = (18,2),name = 'dist_angle')
        y = tf.placeholder(dtype = tf.float32,shape = (None,time_window,1), name = 'y')
        BATCH_SIZE = tf.placeholder(tf.int32)
    l2_regular = tf.contrib.layers.l2_regularizer(0.1)
    with tf.variable_scope('fc1',regularizer = l2_regular):
        fc1_w = tf.Variable(initial_value = tf.random.uniform(shape = (2,fc1_unit),minval = -0.1,maxval = 0.1))
        fc1_b = tf.Variable(initial_value = tf.random.uniform(shape = (fc1_unit,),minval = -0.1,maxval = 0.1))

    with tf.variable_scope('fc2',regularizer = l2_regular):
        fc2_w1 = tf.Variable(initial_value = tf.random_uniform(shape = (HIDDEN_SIZE + fc1_unit,fc2_unit),minval = -0.1,maxval = 0.1))
        fc2_b1 = tf.Variable(initial_value = tf.random.uniform(shape = (fc2_unit,),minval = -0.1,maxval = 0.1))

        fc2_w2 = tf.Variable(
            initial_value = tf.random_uniform(shape = (fc2_unit, fc2_unit), minval = -0.1, maxval = 0.1))
        fc2_b2 = tf.Variable(initial_value = tf.random.uniform(shape = (fc2_unit,), minval = -0.1, maxval = 0.1))

    fc1_mut = tf.nn.relu(tf.matmul(source_dist_angle,fc1_w) + fc1_b)
    fc1_mut = tf.nn.dropout(fc1_mut,keep_prob = 0.9)
    fc1_mut = tf.reshape(tf.tile(fc1_mut,[1,BATCH_SIZE]),(18,BATCH_SIZE,fc1_unit))
    fc1_mut = tf.transpose(fc1_mut,[1,0,2]) #(batch_size,18,fc1_unit)

    # two layer lstm
    def lstm_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias = 1.0)
        return lstm

    with tf.variable_scope('lstm_r',regularizer = l2_regular):
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(LAYER)])

        outputs, states = tf.nn.dynamic_rnn(cell, inputs = source_input, dtype = tf.float32,time_major = False)

    # concat lstm layer and fc layer
    lstm_h = states[1][1]

    lstm_h = tf.reshape(tf.tile(lstm_h,[1,18]),(BATCH_SIZE,18,HIDDEN_SIZE))
    # wrong write lstm_h = tf.transpose(lstm_h,[0,2,1]) #(batch_size,18,fc1_unit)

    fc1_lstm = tf.concat([fc1_mut,lstm_h],axis = -1)
    # two fc layer
    fc1_lstm = tf.reshape(fc1_lstm,(-1,HIDDEN_SIZE + fc1_unit))
    fc2_mut1 = tf.nn.relu(tf.matmul(fc1_lstm,fc2_w1) + fc2_b1)
    fc2_mut1 = tf.nn.dropout(fc2_mut1, keep_prob = 0.9)
    fc2_mut2 = tf.nn.relu(tf.matmul(fc2_mut1,fc2_w2) + fc2_b2)
    fc2_mut2 = tf.nn.dropout(fc2_mut2, keep_prob = 0.9)

    fc2_mut2 = tf.reshape(fc2_mut2,(BATCH_SIZE,-1,fc2_unit))

    # the right of model done
    # build the model left
    with tf.variable_scope('lstm_l',regularizer = l2_regular):
        cell_l = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(LAYER)])

        outputs_l, states_l = tf.nn.dynamic_rnn(cell_l, inputs = target_input_mete, dtype = tf.float32, time_major = False)

    with tf.variable_scope('fc2_l',regularizer = l2_regular):
        fc2_w1_l = tf.Variable(initial_value = tf.random_uniform(shape = (HIDDEN_SIZE,fc2_unit),minval = -0.1,maxval = 0.1))
        fc2_b1_l = tf.Variable(initial_value = tf.random.uniform(shape = (fc2_unit,),minval = -0.1,maxval = 0.1))

        fc2_w2_l = tf.Variable(
            initial_value = tf.random_uniform(shape = (fc2_unit, fc2_unit), minval = -0.1, maxval = 0.1))
        fc2_b2_l = tf.Variable(initial_value = tf.random.uniform(shape = (fc2_unit,), minval = -0.1, maxval = 0.1))

    # two fc layer
    fc1_lstm_l = tf.reshape(states_l[1][1], (-1,HIDDEN_SIZE))
    fc2_mut1_l = tf.nn.relu(tf.matmul(fc1_lstm_l, fc2_w1_l) + fc2_b1_l)
    fc2_mut1_l = tf.nn.dropout(fc2_mut1_l, keep_prob = 0.9)

    fc2_mut2_l = tf.nn.relu(tf.matmul(fc2_mut1_l, fc2_w2_l) + fc2_b2_l)
    fc2_mut2_l = tf.nn.dropout(fc2_mut2_l, keep_prob = 0.9)
    fc2_mut2_l = tf.reshape(fc2_mut2_l, (-1, 1, fc2_unit))

    with tf.variable_scope('attention_layer',regularizer = l2_regular):
        atten_w1 = tf.Variable(initial_value = tf.random_uniform(shape = (HIDDEN_SIZE + fc1_unit,fc2_unit),minval = -0.1,maxval = 0.1))
        atten_b1 = tf.Variable(initial_value = tf.random_uniform(shape = (fc2_unit,),minval = -0.1,maxval = 0.1))
        atten_w2 = tf.Variable(
            initial_value = tf.random_uniform(shape = (fc2_unit,1), minval = -0.1, maxval = 0.1))
        atten_b2 = tf.Variable(initial_value = tf.random_uniform(shape = (1,), minval = -0.1, maxval = 0.1))

    atten_x = tf.tile(fc2_mut2_l,[1,18,1])

    atten_x = tf.concat([fc2_mut2,atten_x],axis = -1)
    atten_x = tf.reshape(atten_x,(-1,atten_x.shape[2]))

    atten_mut1 = tf.nn.relu(tf.matmul(atten_x,atten_w1) + atten_b1)
    atten_mut1 = tf.nn.dropout(atten_mut1,keep_prob = 0.9)
    atten_mut2 = tf.matmul(atten_mut1,atten_w2) + atten_b2
    atten_mut2 = tf.nn.dropout(atten_mut2,keep_prob = 0.9)
    atten = tf.reshape(atten_mut2,(BATCH_SIZE,-1))
    # print(atten.shape)
    a_atten = tf.exp(atten) / tf.reshape(tf.tile(tf.reduce_sum(tf.exp(atten),axis = 1),[18]),(BATCH_SIZE,18))

    a_sum = tf.reshape(tf.tile(a_atten,[1,fc2_unit]),(BATCH_SIZE,fc2_unit,18))
    a_sum = tf.transpose(a_sum,[0,2,1])

    a_sum = tf.multiply(a_sum,fc2_mut2)

    a_sum = tf.reduce_sum(a_sum, axis = 1)
    fc2_mut2_l = tf.squeeze(fc2_mut2_l)

    fusion_x = tf.concat([fc2_mut2_l,a_sum],axis = -1)

    with tf.variable_scope('fusion_layer',regularizer = l2_regular):
        fusion_w1 = tf.Variable(
            initial_value = tf.random_uniform(shape = (HIDDEN_SIZE + fc1_unit, fc2_unit), minval = -0.1, maxval = 0.1))
        fusion_b1 = tf.Variable(initial_value = tf.random_uniform(shape = (fc2_unit,), minval = -0.1, maxval = 0.1))
        fusion_w2 = tf.Variable(
            initial_value = tf.random_uniform(shape = (fc2_unit, time_window), minval = -0.1, maxval = 0.1))
        fusion_b2 = tf.Variable(initial_value = tf.random_uniform(shape = (time_window,), minval = -0.1, maxval = 0.1))

    fusion_mut1 = tf.nn.relu(tf.matmul(fusion_x,fusion_w1) + fusion_b1)
    fusion_mut1 = tf.nn.dropout(fusion_mut1,keep_prob = 0.9)
    y_hat = tf.matmul(fusion_mut1,fusion_w2) + fusion_b2
    y_hat = tf.reshape(y_hat,(BATCH_SIZE,time_window,1))
    l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_sum(tf.square(y - y_hat)) + tf.add_n(l2_loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # feed1 = np.random.random((18,TIME_STEP,32))
    # feed2 = np.random.random((18,2))
    # feed3 = np.random.random((1,TIME_STEP, 31))
    val_error = np.inf
    saver = tf.train.Saver()
    tf.add_to_collection('predict',y_hat)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # random 10 times
        ans = 0;
        for i in range(10):
            val_error = np.inf
            list_s = list(range(36))
            # remove 13 and 20 because they have to much missing value
            list_s.remove(12)
            list_s.remove(19)
            list_s.remove(8), list_s.remove(14)
            list_s.remove(20)
            test_set = random.sample(list_s, 12)

            train_set = set(list_s) ^ set(test_set)
            target_station = random.sample(train_set, 1)
            source_station = train_set ^ set(target_station)

            logger.info('source_station: %s target_station: %s' % (source_station, target_station))
            dp = DataProvider(source_station, target_station, 'PM25_Concentration', time_window)
            scaler = dp.get_scaler()
            n = dp.get_size()
            dist_angle = mergeDistAngle(target_station, source_station)
            dist_angle = dist_angle.reshape(dist_angle.shape[0], dist_angle.shape[1])

            ans_tmp = 0
            for j in range(10):
                for k in range(n // batch_size + 1):
                    source_train,target_train_mete,target_train_aqi = dp.get_batch_data(batch_size)

                    result = sess.run([train_op, y_hat],
                                      feed_dict = {source_input: source_train, target_input_mete: target_train_mete,
                                                   source_dist_angle: dist_angle, y: target_train_aqi,BATCH_SIZE: 64})
                    y_ = np.squeeze(scaler.inverse_transform(result[1]))
                    y_true = np.squeeze(scaler.inverse_transform(target_train_aqi))

                    # logger.info('is equal {}'.format(result[2]))
                    if k % 50 == 0:
                        logger.info('iterator : ' + str(j * n + k * batch_size) + '  train loss is {:.4f}'.format(
                            np.sqrt(mean_squared_error(y_true, y_))))
                        source_val,target_val_mete,target_val_aqi = dp.validation()
                        result = sess.run(y_hat,
                                         feed_dict = {source_input: source_val, target_input_mete: target_val_mete,
                                                      source_dist_angle: dist_angle, y: target_val_aqi,BATCH_SIZE:875})

                        y_ = np.squeeze(scaler.inverse_transform(result))
                        y_true = np.squeeze(scaler.inverse_transform(target_val_aqi))
                        error = np.sqrt(mean_squared_error(y_true, y_))
                        logger.info('iterator : ' + str(j * n + k * batch_size) + '  val loss is {:.4f}'.format(np.sqrt(mean_squared_error(y_true, y_))))
                        if val_error > error:
                            val_error = error
                            saver.save(sess, 'model/my-model', global_step = j * n + k * batch_size)
                            # test
                            source_test, target_test_mete, target_test_aqi = dp.test()

                            result = sess.run([y_hat,a_atten],
                                              feed_dict = {source_input: source_test, target_input_mete: target_test_mete,
                                                           source_dist_angle: dist_angle,BATCH_SIZE:877})

                            y_ = np.squeeze(scaler.inverse_transform(result[0]))
                            y_true = np.squeeze(scaler.inverse_transform(target_test_aqi))
                            ans_tmp = np.sqrt(mean_squared_error(y_true, y_))
                            pd.DataFrame(result[1]).to_csv('attention' + str(i) + '.csv')
                            logger.info('station {} test loss is {:.4f}'.format(target_station, np.sqrt(mean_squared_error(y_true, y_))))
            ans += ans_tmp
        logger.info('average rmse of random 10 is : {:.4f}'.format(ans / 10))


if __name__ == '__main__':
    Main()
    # arr = np.array([[1,2,3],[4,5,6]])
    # arr = arr.reshape((2,1,3))
    # # print(np.repeat(arr,repeats = 2,axis = 1))
    # repeat = np.repeat(arr, repeats = 4, axis = 1)
    # print(repeat)
    # print(repeat[:,0,:],repeat[:,1,:])

    # lstm_h = tf.constant([[1,2,3],[4,5,6]])
    # lstm_h = tf.reshape(tf.tile(lstm_h, [1, 18]), (2, 18, 3))
    # # lstm_h = tf.transpose(lstm_h, [0, 2, 1])  # (batch_size,18,fc1_unit)
    #
    # equal = tf.equal(lstm_h[:, 0, :], lstm_h[:, 1, :])
    # with tf.Session() as sess:
    #     result = sess.run(lstm_h)
    #     print(result[:,0,:],result[:,1,:],result.shape)