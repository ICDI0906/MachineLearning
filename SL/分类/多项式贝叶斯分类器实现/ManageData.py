# 将数据进行整理，分为训练数据90% 和测试数据10%

import pandas as pd
import os
dir = '/Users/icdi/Desktop/3'
train_text = []
train_label = []
test_text = []
test_label = []
name_to_index = {}
prop = 0.9
frame = pd.read_excel(dir+'/index_name.xls')

for key in frame.index:
    values = frame.loc[key].values
    name_to_index[values[1]] = values[0]

for file in os.listdir(dir):
    if(file.endswith('.txt')):
        name = file[:file.index('_')]
        name_content = open(dir + '/' + file).read().split('\n')[:-1]

        trainprop = int(prop*len(name_content))
        train_text.extend(name_content[:trainprop])
        test_text.extend(name_content[trainprop:])

        label = [str(name_to_index[name]) for i in range(len(name_content))]
        train_label.extend(label[:trainprop])
        test_label.extend(label[trainprop:])

# 将文本进行保存
train_text_file = open('train_text.txt','w')
train_text_file.write('\n'.join(train_text))

test_text_file = open('test_text.txt','w')
test_text_file.write('\n'.join(test_text))

train_label_file = open('train_label.txt','w')
train_label_file.write('\n'.join(train_label))

test_label_file = open('test_label.txt','w')
test_label_file.write('\n'.join(test_label))