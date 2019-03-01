# @Time : 2019/3/1 8:40 PM 
# @Author : Kaishun Zhang 
# @File : test.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
import re
s = '\ndsd\t\n'
pattern = '[\n\t]+'
print(re.split(pattern,s))