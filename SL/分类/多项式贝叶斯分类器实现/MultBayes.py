# 理解贝叶斯分类器
# 贝叶斯分类器一般分为3种
# 一个伯努利，一个高斯，一个多项式。这样分主要是根据先验分布来说的
# 如果假设先验分布为高斯分布，这可使用Gussis分布
# 具体可见 讲解https://www.cnblogs.com/pinard/p/6074222.html
from collections import defaultdict
class MultBayes(object):
    P = {}  # 每一类的概率
    N = 0   # 样本数量
    vocabulary = defaultdict(int)  # 单词所对对应的索引


