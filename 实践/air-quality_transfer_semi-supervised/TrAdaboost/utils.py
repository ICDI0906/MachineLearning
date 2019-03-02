# @Time : 2019/3/2 10:01 AM 
# @Author : Kaishun Zhang 
# @File : load_data.py 
# @Function:
import numpy as np
from sklearn.datasets import fetch_20newsgroups
# 20newsgroup 是按照时间序列进行分割测试集和训练集的
rec = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
talk = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
ng_train_rec = fetch_20newsgroups(subset = 'train', categories = rec)
ng_train_talk = fetch_20newsgroups(subset = 'train', categories = talk)
ng_test_rec = fetch_20newsgroups(subset = 'test', categories = rec)
ng_test_talk = fetch_20newsgroups(subset = 'test', categories = talk)


def load_rec_talk():
    ng_rec = ng_train_rec.data + ng_test_rec.data
    ng_talk = ng_train_talk.data + ng_test_talk.data
    ng_rec_target = np.concatenate((ng_train_rec.target, ng_test_rec.target))
    ng_talk_target = np.concatenate((ng_train_talk.target, ng_test_talk.target))

    return ng_rec, ng_talk, ng_rec_target, ng_talk_target


def load_rec_talk_mix():
    # res 为 positive talk 为negative
    ng_all_data = ng_train_rec.data + ng_train_talk.data + ng_test_rec.data + ng_test_talk.data
    minus_one = np.zeros(len(ng_train_talk.data)) - 1;
    minus_two = np.zeros(len(ng_test_talk.data)) - 1;
    ng_all_target = np.concatenate((np.ones(len(ng_train_rec.data)),minus_one,np.ones(len(ng_test_rec.data)),minus_two))
    return ng_all_data,ng_all_target


def score_acc(clf,X_test,y_test):
    pred_y = clf.predict(X_test)
    return 1 - np.mean(pred_y == y_test)
