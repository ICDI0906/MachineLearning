# @Time : 2018/10/8 上午10:42 
# @Author : Kaishun Zhang 
# @File : coin.py 
# @Function: 使用EM 算法来得到估计硬币朝上的概率
import numpy  as np
pi = 0.5; p = 0.5; q = 0.5


def eStep():
    pass

def train(data):
    # data  = np.array([1,1,0,1,0,0,1,0,1,1])
    print(data)
    global pi,p,q
    while(True):
        pi_old = pi;p_old = p;q_old = q
        u_j = []   # 对于第j个样例，来自B的概率，那么来自C 的概率就是 1 - u_j
        # 这就是E步
        for y_j in data:
            u_j.append((pi_old * pow(p_old , y_j) * pow(1 - p_old, 1 - y_j))/ (pi_old * pow(p_old , y_j) * pow(1 - p_old, 1 - y_j) +
                                                                            ((1-pi_old) * pow(q_old,y_j) * pow(1 - q_old, 1 - y_j))))
        # 反过来来更新参数
        u_sum = np.sum(np.array(u_j))
        pi = u_sum / len(u_j)
        p = np.sum(np.array(u_j) * data) / u_sum
        q = np.sum((1 - np.array(u_j)) * data) / u_sum
        print(pi, p, q)
        if(pi == pi_old and p == p_old and q == q_old):
            break


def generateData(size = 10):
    data = []
    for i in range(size):
        data.append(round(np.random.rand()))
    return np.array(data)


if __name__ == '__main__':
    data = generateData()
    train(data)