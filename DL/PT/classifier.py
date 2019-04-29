# @Time : 2019/4/29 3:11 PM 
# @Author : Kaishun Zhang 
# @File : classifier.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
n_data = torch.ones(100,2)

x0 = torch.normal(2 * n_data, 1)

y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
## 每一个采样的均值和方差, mean 和 std的大小不一定相同

y1 = torch.ones(100)

x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)

net = Net(2,100,2)
optimizer = torch.optim.SGD(net.parameters(),lr = 0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        ## 返回的最大值，还有对应最大的所在的坐标
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c = pred_y, s= 100,lw = 0,cmap = 'RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict = {'size': 20, 'color': 'red'})
        plt.pause(0.1)