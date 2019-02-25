# @Time : 2018/11/28 下午9:50 
# @Author : Kaishun Zhang 
# @File : knn.py
# @Function:
from numpy.linalg import *
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score
class Node(object):
    def __init__(self,dim,data,label): # 按照哪个维度进行划分的，数据，还有标记
        self.dim = dim
        self.data = data
        self.label = label
        self.left = None
        self.right = None

    def __str__(self):
        print('dim : ',self.dim,' data: ',self.data, 'label : ',self.label)

eps = 1e-12
class KNN(object):
    def __init__(self):
        pass
    def train(self,train_data,train_label):
        self.k = train_data.shape[1]
        row_index = train_data[:,0].argsort()
        train_data = train_data[row_index]
        train_label = train_label[row_index]
        middle_index = train_data.shape[0] // 2
        self.root = Node(0,train_data[middle_index],train_label[middle_index])
        self.root.left = self.build_tree(train_data[:middle_index],train_label[:middle_index],1)
        self.root.right = self.build_tree(train_data[middle_index + 1:],train_label[middle_index + 1:],1)

    def build_tree(self,train_data,train_label,dep):
        index = dep % self.k
        row_index = train_data[:, index].argsort()
        train_data = train_data[row_index]
        train_label = train_label[row_index]
        middle_index = train_data.shape[0] // 2
        # print(train_data[middle_index],'label start --- ',train_label[middle_index],'label end ---- ')
        root = Node(index, train_data[middle_index], train_label[middle_index])
        if train_data.shape[0] == 1:
            return root
        else:
            if train_data[:middle_index].shape[0] > 0:
                root.left = self.build_tree(train_data[:middle_index],train_label[:middle_index],dep + 1)
            if train_data[middle_index + 1:].shape[0] > 0:
                root.right = self.build_tree(train_data[middle_index + 1:], train_label[middle_index + 1:],dep + 1)
            return root

    def predict(self,test_data):
        root = self.root
        self.predict_dist = 1e9
        self.predict_label = []
        self.test_data = test_data
        self._predict(root)

    def _predict(self,root):
        if root.left == None and root.right == None:  # when it is a leaf, compare
            if self.predict_dist > self.dist(root.data):
                self.predict_dist = self.dist(root.data)
                self.predict_label = root.label
            # elif abs(self.predict_dist - self.dist(root.data)) < eps:
            #     self.predict_label.append(root.label)
            return
        if root.data[root.dim] > self.test_data[root.dim]: # in left
            if not root.left is None:
                self._predict(root.left)
            if abs(root.data[root.dim] - self.test_data[root.dim]) < self.predict_dist:  # intersect with the dim
                # judge the root data
                if self.predict_dist > self.dist(root.data):
                    self.predict_dist = self.dist(root.data)
                    self.predict_label = root.label
                # elif abs(self.predict_dist - self.dist(root.data)) < eps:
                #     self.predict_label.append(root.label)

                if not root.right is None:
                    self._predict(root.right)

        elif root.data[root.dim] < self.test_data[root.dim]: # in right
            if not root.right is None:
                self._predict(root.right)
            if abs(root.data[root.dim] - self.test_data[root.dim]) < self.predict_dist:
                # judge the root data
                if self.predict_dist > self.dist(root.data):
                    self.predict_dist = self.dist(root.data)
                    self.predict_label = root.label
                # elif abs(self.predict_dist - self.dist(root.data)) < eps:
                #     self.predict_label.append(root.label)
                if not root.left is None:
                    self._predict(root.left)
        else:
            if not root.left is None:
                self._predict(root.left)
            if not root.right is None:
                self._predict(root.right)
        return

    def dist(self,lista):
        return norm(lista - self.test_data)

    def dfs(self):
        root = self.root
        self._dfs(root, 0)

    def _dfs(self,root,dep):
        if root == None:
            return
        print('root.dim',root.dim,'root.data ',root.data,'label : ',root.label ,'dep ',dep)
        self._dfs(root.left,dep + 1)
        self._dfs(root.right,dep + 1)


if __name__ == '__main__':
    knn = KNN()
    data_path = 'avila-tr.csv'
    frame = pd.read_csv(data_path)
    print(frame.shape)
    train_data = frame.values[:,:-1]
    train_label = frame.values[:,-1]
    start = time.time()
    knn.train(train_data,train_label)
    end = time.time()
    print('train cost {0} s'.format((end - start)))
    # print(knn.root)
    frame = pd.read_csv('avila-tr.csv')
    test_data = frame.values[:,:-1]
    test_label = frame.values[:,-1]
    predict_lable = []
    result = defaultdict(int)
    for i,test in enumerate(test_data):
        knn.predict(test)
        # for item in knn.predict_label:
        #     result[item]+=1
        # result_sorted = sorted(result.items(),key = lambda x:x[1],reverse = True)
        # if result_sorted[0][1] == 1: # 出现所有的概率相同，随机选择一个输出
        #     predict_index = np.random.randint(0,len(result_sorted))
        #     predict = result_sorted[predict_index][0]
        #
        # else:
        #     predict = result_sorted[0][0];
        predict = knn.predict_label
        print(predict,' ----  ',test_label[i])
        predict_lable.append(predict)
    # print(predict_lable)
    core = accuracy_score(test_label,predict_lable)
    print('accuracy : ',core)
#   accuracy :  0.9546897546897547
    # vaila-tr 0.6985435032579532
