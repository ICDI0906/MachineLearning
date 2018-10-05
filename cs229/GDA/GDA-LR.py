# @Time : 2018/10/5 下午2:23 
# @Author : Kaishun Zhang 
# @File : GDA-LR.py 
# @Function: 验证一下gda 和 逻辑回归之间的关系
# p(y=1|x) =
# (p(x|y=1) * p(y=1)) / (p(x|y=0)*p(y=0) + p(x|y=1)*p(y=1))
import numpy as np
import matplotlib.pyplot as plt


def Gaussian(x , u =10, sigma = 2):
    return 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - u)**2 / (2 * sigma ** 2))


def py1givenx(x):
    return Gaussian(x ,15 ,2) * 0.5 / (Gaussian(x, 10 ,2) * 0.5 + Gaussian(x ,15, 2) * 0.5)


def main():
    sigma = 2
    X1 = np.random.normal(10, 2 ,30)
    X1 = np.array(sorted(X1))
    plt.plot(X1, 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (X1 - 10)**2 / (2 * sigma ** 2)), linewidth = 2, color = 'r')
    X2 = np.random.normal(15 ,2 ,30)
    X2 = np.array(sorted(X2)) # need to change
    plt.plot(X2, 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(X2 - 15)**2 / (2 * sigma ** 2)), linewidth = 2, color = 'b')
    predX = np.linspace(0,20,200)
    predY = list(map(py1givenx,predX))
    plt.plot(predX,predY ,linewidth = 2, color = 'g')
    # plt.show()
    plt.title("GDA and logistic regression")
    plt.savefig('gda-lr.jpg')


if __name__ == '__main__':
    main()