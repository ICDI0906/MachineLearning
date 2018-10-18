# @Time : 2018/10/17 下午7:33 
# @Author : Kaishun Zhang 
# @File : distribution.py 
# @Function: 概率论中常见的分布
import matplotlib.pyplot as plt
import numpy as np
# 计算n 的阶乘
def factorial(n):
    if n == 1 or n == 0:
        return 1
    else:
        return n * factorial(n-1)
# 泊松分布
def poisson(lamda):
    x = range(15)
    y = []
    for i in x:
        y.append(np.exp(-lamda) * np.power(lamda,i) / factorial(i))
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x,y)
    plt.xticks([0,15])
    # 或者也可以这样,只是这样并不是概率，而是采样
    y = np.random.poisson(lamda,15)
    print(y)
    plt.subplot(2,2,2)
    plt.plot(x,y)
    plt.show()

# 指数分布
def exponential(lamda):
    x = np.array(range(15))
    # y = np.random.exponential(lamda,15)
    y = lamda * np.exp(-lamda * x)
    plt.plot(x,y)
    plt.show()

# gamma 分布

def logGamma(x):
      tmp = (x - 0.5) * np.log(x + 4.5) - (x + 4.5);
      ser = 1.0 + 76.18009173/ (x + 0)- 86.50532033/(x + 1) + 24.01409822/ (x + 2)-1.231739516/ (x + 3) \
            + 0.00120858003/ (x + 4)-0.00000536382 /(x + 5);
      return tmp + np.log(ser * np.sqrt(2 * np.pi));


def gamma(lamda,alpha):
    x = np.linspace(0,12,50)
    gamma_alpha = np.exp(logGamma(alpha))
    y = lamda * np.exp(-lamda * x) * np.power(lamda * x, alpha - 1) / gamma_alpha
    plt.plot(x,y)
    plt.savefig('gamma.jpg')


if __name__ == '__main__':

    gamma(1,0.5)
    gamma(1,2.3)
    gamma(1,3.4)
    gamma(1,5)
    plt.show()