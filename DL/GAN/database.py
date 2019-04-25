# @Time : 2019/4/25 10:28 AM 
# @Author : Kaishun Zhang 
# @File : database.py 
# @Function:
import numpy as np
import matplotlib.image as img
import matplotlib.pylab as plt
import os


class db(object):
    def __init__(self):
        path = '/Users/icdi/Downloads/faces/'
        self.all_data = []
        for file in os.listdir(path):
            i = img.imread(path + file)
            self.all_data.append(i)
        self.size = len(self.all_data)

    def get_data(self):
        return np.array(self.all_data)[:100];

#
# db_generator = db()
# data = db_generator.get_batch_data(100)
# img.imread(data[0])
# plt.imshow(data[0])
# plt.show()