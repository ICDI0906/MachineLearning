# @Time : 2018/10/15 上午10:41 
# @Author : Kaishun Zhang 
# @File : quantile_plot.py 
# @Function: 分位图
import numpy  as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import scipy.stats as stats

def get_data():
    x = [40,43,47,74,75,78,115,117,120]
    return np.array(x)

# quanties-plot for univariate


def plot_quantile():
    y = get_data();
    x = [(i - 0.5) / len(y) for i in range(len(y))]
    plt.scatter(x,y)
    plt.xticks([0.0,0.25,0.5,0.75,1.0])
    plt.show()

# 但是感觉无法自己去定义dst的分布


def plot_quantile_quantile():
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    res = mod_fit.resid
    print(res)
    # fig = sm.qqplot(res)
    # plt.show()
    # t 分布， 4个自由度 拟合 线的夹角是45度
    fig = sm.qqplot(res, stats.t, distargs = (4,), fit = True, line='45')
    plt.show()



if __name__ == '__main__':
    # plot_quantile()
    # plot_quantile_quantile()
    plot_quantile_quantile()
