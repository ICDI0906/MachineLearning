# @Time : 2019/3/25 3:26 PM 
# @Author : Kaishun Zhang 
# @File : rnn-tensorflow.py 
# @Function: 主要是RNN.ipynb进行一个对比


from keras.layers import Dense,Activation,SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from time import time

from tensorflow.examples.tutorials.mnist import input_data
mn = input_data.read_data_sets('MNIST_DATA',one_hot = True)


TIME_STEP = 28
CELL_SIZE = 400
INPUT_SIZE = 28

OUTPUT_SIZE = 10
BATCH_SIZE = 1000
model = Sequential()
model.add(SimpleRNN(
    batch_input_shape = [None,TIME_STEP,INPUT_SIZE],
    output_dim = CELL_SIZE,
    unroll = False
))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
Iterator = 1000
adam = Adam(lr = 1e-3)
#### Adam 收敛速度快，SGD虽然收敛，但是慢
model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
start = time()
for i in range(Iterator):
    train_data, train_label = mn.train.next_batch(BATCH_SIZE)
    train_data = train_data.reshape(-1,TIME_STEP,INPUT_SIZE)
    model.train_on_batch(train_data,train_label)
    if i % 10 == 0:
        loss, acc = model.evaluate(train_data, train_label)
        print('loss is {} and the accuracy is {}'.format(loss, acc))

test_x,test_y = mn.test.next_batch(BATCH_SIZE)
loss, acc = model.evaluate(test_x, test_y)
print(loss,acc)
print(time() - start)
