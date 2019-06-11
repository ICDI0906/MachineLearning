# @Time : 2019/6/8 2:24 PM 
# @Author : Kaishun Zhang 
# @File : utils.py 
# @Function:
import math
import logging,os,sys
import numpy as np
import pandas as pd
import time
import random



class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    def __init__(self,min,max):
        self.min = min
        self.max = max

    def transform(self,data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self,data):
        return data * (self.max - self.min) + self.min


def getDistance(long1,  lat1,  long2, lat2):
    R = 6378137 # 地球半径
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    a = lat1 - lat2
    b = (long1 - long2) * math.pi / 180.0
    sa2 = math.sin(a / 2.0)
    sb2 = math.sin(b / 2.0)
    d = 2 * R * math.asin(math.sqrt(sa2 * sa2 + math.cos(lat1) * math.cos(lat2) * sb2 * sb2))
    return d


def getAngle(lat_a, lng_a, lat_b, lng_b):
    y = math.sin(lng_b - lng_a) * math.cos(lat_b)
    x = math.cos(lat_a) * math.sin(lat_b) - math.sin(lat_a) * math.cos(lat_b) * math.cos(lng_b - lng_a)

    brng = math.atan2(y, x)

    brng = brng * 180 / math.pi
    if brng < 0:
        brng = brng + 360
    return brng


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

logger = get_logger('.',name = 'utils')


def mergeMeteData(station):
    '''
    :param station: station id
    :return: train data,val data
    '''
    sta_list = list(station)
    frame_list = []
    for sta in sta_list:
        if sta >= 9:
            filename = 'mete_10' + str(sta + 1) + '_norm.csv'
        else:
            filename = 'mete_100' + str(sta + 1) + '_norm.csv'
        frame = pd.read_csv(filename)
        frame.drop(['id','time'],axis = 1, inplace = True)
        frame_list.append(frame.values)

    return np.concatenate(frame_list, axis = 1)


def mergeAqiData(station,type):
    '''
    :param station: station id
    :param type: 'PM2.5' etc
    :return: train_data,val_data, scaler for decomposition
    '''
    sta_list = list(station)
    frame_list = []
    for sta in sta_list:
        if sta >= 9:
            filename = 'aqi_10' + str(sta + 1) + '.csv'
        else:
            filename = 'aqi_100' + str(sta + 1) + '.csv'
        frame = pd.read_csv(filename)
        frame_tmp = frame[type]
        frame_tmp.fillna(frame_tmp.mean(),inplace = True)
        scaler = MinMaxScaler(min = frame_tmp.min(),max = frame_tmp.max())
        frame_tmp = scaler.transform(frame_tmp)
        frame_list.append(frame_tmp.values.reshape(frame_tmp.values.shape[0],1))
    all_data = np.concatenate(frame_list, axis = 1)
    if len(sta_list) == 1:
        return all_data, scaler
    return all_data


def mergeDistAngle(target_station,source_station):
    '''
    :param target_station: station id
    :param source_station: station id
    :return: distance and angle relative to target
    '''
    sta_list = list(source_station)
    target = list(target_station)[0]
    dist_frame = pd.read_csv('dist.csv')
    angle_frame = pd.read_csv('angle.csv')
    result = []
    for i in sta_list:
        tmp = []
        tmp.append(dist_frame.values[i,target])
        tmp.append(angle_frame.values[target,i])
        result.append(tmp)
    return np.array(result)


class DataProvider(object):
    def __init__(self, source_station, target_station, type, time_window, train_percentage = 0.8, val_percentage = 0.1, num_classes = 1):
        logger.info('load data ....')

        start = time.time()
        self.source_mete = mergeMeteData(source_station)
        self.source_aqi = mergeAqiData(source_station,type = type)
        self.source = np.concatenate([self.source_mete,self.source_aqi],axis = 1)
        self.target_mete = mergeMeteData(target_station)
        self.target_aqi,self.scaler = mergeAqiData(target_station,type = type)

        logger.info('load data cost %s' % (time.time() - start))

        self.n = self.source.shape[0]
        self.source_dim = self.source.shape[1]
        self.target_mete_dim = self.target_mete.shape[1]
        self.target_aqi_dim = self.target_aqi.shape[1]

        self.time_window = time_window
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.num_classes = num_classes

        # maybe we should write it like this
        train_pos = int(self.train_percentage * self.n)
        val_pos = int(self.val_percentage * self.n) + train_pos

        self.train_index = list(range(0, train_pos - self.time_window + 1))
        self.val_index = list(
            range(train_pos - self.time_window + 1, val_pos - self.time_window + 1))
        self.test_index = list(range(val_pos - self.time_window + 1, self.n - self.time_window + 1))

        logger.info(
            "\nSplit data finished!\n"
            "total sample :{} ,train: {}, val: {}, test :{} ".format(
                self.n, len(self.train_index), len(self.val_index),len(self.test_index)
            )
        )

    def get_size(self):
        return self.n

    def get_scaler(self):
        return self.scaler

    def _generate_batch(self, index_list):
        il = len(index_list)
        source = np.zeros([il,self.time_window,self.source_dim])
        target_mete = np.zeros([il,self.time_window,self.target_mete_dim])
        target_aqi = np.zeros([il,self.time_window,self.target_aqi_dim])

        for i,index in enumerate(index_list):
            source[i,:,:] = self.source[index:index + self.time_window,:]
            target_mete[i,:,:] = self.target_mete[index:index + self.time_window, :]
            target_aqi[i,:,:] = self.target_aqi[index:index + self.time_window,:]

        return source,target_mete,target_aqi

    def get_batch_data(self, batch_size):
        index_list = random.sample(self.train_index, batch_size)
        return self._generate_batch(index_list)

    def validation(self):
        return self._generate_batch(self.val_index)

    def test(self):
        return self._generate_batch(self.test_index)