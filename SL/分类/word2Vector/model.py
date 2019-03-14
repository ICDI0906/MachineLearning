# # learn from https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html
# 该代码是为了通过神经网络K-grams获得
# 当输入一个词时，在词典中找出最相近的词

import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter
with open('mingci.txt',encoding='utf-8') as f:
    text = f.read()
words=[]
# for data in text.split('\n'):
#     for tmp in data.split(' '):
#         if len(tmp.strip()):
#             words.append(tmp)
for data in text.split(' '):
        if len(data.strip()):
            words.append(data.strip())
words_count = Counter(words)
words = [w for w in words if words_count[w] > 10]
vocab = set(words)
word_to_int = {w:d for d,w in enumerate(vocab)}
int_to_word = {d:w for d,w in enumerate(vocab)}

# print("total words: {}".format(len(words)))
# print("unique words: {}".format(len(set(words))))
int_words = [word_to_int[w] for w in words]


t = 1e-5 # t值
threshold = 0.9  # 要删除率大于90%的

int_words_counter = Counter(int_words)


total_count = len(int_words)
int_words_fre = {w:f/total_count for w, f in int_words_counter.items()}
int_words_drop = {w:1-np.sqrt(t/fre) for w, fre in int_words_fre.items()}
train_word = [w for w in int_words if int_words_drop[w] < threshold]
# print(len(train_word))


def get_targets(words, idx, window_size=5):
    target_window = np.random.randint(1,window_size+1)
    start_point = idx-target_window if idx-target_window >=0 else 0
    end_point = idx+target_window
    target_set = set(words[start_point:idx]+words[idx:idx+end_point+1])
    return list(target_set)


def get_batches(words, batch_size, window_size=5):
    n_batch = len(words)//batch_size
    words = words[: (n_batch*batch_size)]
    for idx in range(0,len(words),batch_size):
        x = [];y=[]
        data = words[idx:idx+batch_size]
        for i in range(len(data)):
            target_x = data[i]
            target_y = get_targets(data,i,window_size=window_size)
            x.extend([target_x]*len(target_y))
            y.extend(target_y)
        yield x,y


train_graph = tf.Graph()
with train_graph.as_default():
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

vocab_size = len(int_to_word)
embedding_size = 200

with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1)) #产生-1 -- 1 均匀分布的数据
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)

n_sampled = 100
with train_graph.as_default():
    with tf.name_scope('fc0'):
        softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(vocab_size))
    # 计算negative sampling下的损失
    with tf.name_scope('loss'):
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        cost = tf.reduce_mean(loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    # 随机挑选一些单词
    ## From Thushan Ganegedara's implementation
    valid_size = 7  # Random set of words to evaluate similarity on.
    valid_window = 100
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2)) #在range 范围中随机选择一些点
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    with tf.name_scope('valid_example'):
        valid_examples = [word_to_int['马云'],
                          word_to_int['百度'],
                          word_to_int['山东']]
    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))# 求平方之后，然后按照列进行求和运算
    normalized_embedding = embedding / norm # 数据除以标准差
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding)) #
    # [valid_size * embedding_size] *  [embedding_size * vocab] 相似度矩阵

epochs = 10  # 迭代轮数
batch_size = 1000  # batch大小
window_size = 5  # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver()  # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_word, batch_size, window_size)
        start = time.time()
        #
        for x, y in batches:
            # print(x,y)
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 50 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # 计算相似的词
            print(iteration)
            if iteration % 50 == 0:
                # 计算similarity
                sim = similarity.eval() # 返回的是numpy.darray() 类型的数组
                for i in range(valid_size):
                    valid_word = int_to_word[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1] # 返回的是 索引
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_word[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)
    writer = tf.summary.FileWriter(logdir="./log",graph=sess.graph)