# @Time : 2018/11/30 上午8:48 
# @Author : Kaishun Zhang 
# @File : knn_lib.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
nc = KNeighborsClassifier()

data_path = 'avila-tr.csv'
frame = pd.read_csv(data_path)
train_data = frame.values[:, :-1]
train_label = frame.values[:, -1]
start = time.time()
nc.fit(train_data,train_label)
end = time.time()
print('train cost {0} s'.format(end - start))
data_path = 'avila-ts.csv'
frame = pd.read_csv(data_path)
test_data = frame.values[:, :-1]
test_label = frame.values[:, -1]

predict_label = nc.predict(test_data)
# print(predict_label[0])
print('accuracy is {0}'.format(accuracy_score(test_label,predict_label)))
