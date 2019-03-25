import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from time import time
tf.set_random_seed(1)
mnist = input_data.read_data_sets("MNIST_DATA",one_hot = True)

lr = 0.01
training_iter = 2000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_unit = 128
n_classes = 10

x = tf.placeholder(tf.float32,shape = [None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,shape = [None,n_classes])

weigths = {
    "in":tf.Variable(tf.random_normal([n_inputs,n_hidden_unit])),
    "out": tf.Variable(tf.random_normal([n_hidden_unit,n_classes]))
}
biases = {
    "in":tf.Variable(tf.constant(0.1, shape = [n_hidden_unit,])),
    "out":tf.Variable(tf.constant(0.1, shape = [n_classes,]))
}

def RNN(X, weights, biases):
    # X.shape = n_batch,n_steps,n_inputs
    X = tf.reshape(X, [-1,n_inputs])
    # X.shape = [batch_size *28, 28]
    X_in = tf.matmul(X, weights["in"])+ biases["in"]
    # X,shape = [batch_size,n_steps,n_hidden_unit]
    X_in = tf.reshape(X_in , shape = [-1, n_steps, n_hidden_unit])  #(n_batch, 28,128 hidden_size)

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unit,forget_bias = 1.0, state_is_tuple = True)

    _init_state =lstm_cell.zero_state(batch_size, dtype = tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell , X_in, initial_state = _init_state, time_major =False)

    # result = tf.matmul(states[-1], weights['out']) + biases['out']
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results;

start = time()
pred = RNN(x, weigths, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step < training_iter:
        train_x,train_y = mnist.train.next_batch(batch_size)
        train_x = train_x.reshape([batch_size, n_steps,n_inputs])
        sess.run([train_op],feed_dict = {
            x:train_x,
            y:train_y
        })

        if step % 20 ==0:
            print(sess.run([accuracy],feed_dict = {
                x: train_x,
                y: train_y
            }))
        step += 1

print(time() - start)