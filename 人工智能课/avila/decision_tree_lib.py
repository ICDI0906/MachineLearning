# @Time : 2018/11/29 上午9:08 
# @Author : Kaishun Zhang 
# @File : decision_tree_lib.py 
# @Function:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtc = DecisionTreeClassifier(criterion = "entropy")

data_path = 'avila-tr.csv'
frame = pd.read_csv(data_path)
train_data = frame.values[:, :-1]
train_label = frame.values[:, -1]
start = time.time()
dtc.fit(train_data,train_label)
end = time.time()
print('train cost {0} s'.format(end - start))
data_path = 'avila-ts.csv'
frame = pd.read_csv(data_path)
test_data = frame.values[:, :-1]
test_label = frame.values[:, -1]

predict_label = dtc.predict(test_data)
# print(predict_label[0])
print('accuracy is {0}'.format(accuracy_score(test_label,predict_label)))
