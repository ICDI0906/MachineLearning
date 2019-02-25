# @Time : 2018/11/1 上午11:06 
# @Author : Kaishun Zhang 
# @File : hot_word.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import os

# path_of_font = os.path.join(os.path.dirname("/"), "DroidSansFallbackFull.ttf")

# text_from_file_with_path = open('/Users/heiqie/cloud/heiqie.txt').read()
dic = dict()
with open('hot_word', 'r') as fw:
    data = fw.read().split('\n')
for j in data:
    dic[(j.split('\t')[0])] = int(j.split('\t')[1])

my_wordcloud = WordCloud(font_path="/System/Library/Fonts/PingFang.ttc",max_font_size=60,
                         background_color = "white",  # 背景颜色
                         max_words = 2000,  # 词云显示的最大词数
                         mask = None,  # 设置背景图片
                         ).generate_from_frequencies(dic)

plt.imshow(my_wordcloud)

plt.axis("off")
plt.savefig('tmp.jpg')
# plt.show()
