import numpy as np
# 学习网址：https://www.jianshu.com/p/679e390f24bb
# 未测试，算法推到见图片
class BPNetwork(object):
    def __init__(self, sizes):
        # sizes = [2,3,2]
        # 定义层数
        self.layers = len(sizes)
        # 多少个输入层，中间层和隐藏层
        self.sizes = sizes
        # bias
        self.bias = [np.random.randn(x, 1) for x in sizes[1:]]
        # weights
        self.weights = [np.random.randn(y, x) for (x,y) in zip(sizes[:-1], sizes[1:])]

    def sigmod(self, x):
        return 1.0/(1.0+np.exp(-x))


    # 前向传播
    def forward(self, a):

        # 一次传递数值就可以得到左后的预测值
        for (w, b) in zip(self.weights,self.sizes):
            a = self.sigmod(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        # mini_batch_size 小样本数量
        # eta 学习效率
        # epochs 迭代次数
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            # 每次训练的数据集的多少
            mini_batch_data = [training_data[k, k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_trian in mini_batch_data:
                self.update_mini_batch(mini_trian, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} Done".format(i))

    def backprop(self, x, y):
        nable_b = [np.zeros(b.shape) for b in self.bias]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        # 使用激活函数
        activation = x;
        # 没有使用激活函数列表
        notactivations = []
        # 已经使用激活函数列表

        activations = [x]
        for w , b in zip(self.weights, self.bias):
            notactivation = np.dot(w, activation) + b
            notactivations.append(notactivation)
            activation = self.sigmod(notactivation)
            activations.append(activation)

        delta = (activations[-1] - y) * (1 - self.sigmod(notactivations[-1])) * self.sigmod(notactivations[-1])
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(self.layers):
            delta = np.dot(self.weights[-l+1], delta) * self.sigmod(notactivations[-l]) * (1 - self.sigmod(notactivations[-l]))
            nable_b[-l] = delta
            nable_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nable_b,nable_w)

    def update_mini_batch(self, mini_train, eta):
        nable_b = [np.zeros(b.shape) for b in self.bias]
        nable_w = [np.zeros(w.shape) for w in self.weights]

        for x, y  in mini_train:
            delta_nable_b, delta_nable_w = self.backprop(x, y)
            nable_b = [a+b for a,b in zip(nable_b,delta_nable_b)]
            nable_w = [a+b for a,b in zip(nable_w,delta_nable_w)]
        self.weights = [w - eta/len(mini_train) * w_ for w,w_ in zip(self.weights,nable_w)]
        self.bias =    [b- eta/len(mini_train) * b_ for b,b_ in zip(self.bias,nable_b)]

