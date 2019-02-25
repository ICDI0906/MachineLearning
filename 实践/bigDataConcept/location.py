# @Time : 2018/11/2 下午10:34 
# @Author : Kaishun Zhang 
# @File : location.py 
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
import folium
import jieba
import requests
import json
import time
from scipy.misc import imread
def get_pos():
     place_list = open('location',encoding='utf-8').read()
     place_list = place_list.split('\n')
     ak = '9OlGVUqDCAkAzSrhOgiaTAZ45jNOGdHQ'
     file = open('place.json','w',encoding='utf-8')
     result = []
     for place_pair in place_list:
          place = place_pair.split('\t')[0]
          cnt = int(place_pair.split('\t')[1])
          tmp_result = []
          # time.sleep(0.5)
          url = 'http://api.map.baidu.com/geocoder/v2/?address=' + place + '&output=json&ak=' + ak
          json_data = requests.get(url=url).json()
          # print(json_data)
          if json_data['status'] == 1:
               print(place)
               continue
          tmp_result.append(json_data['result']['location']['lat'])
          tmp_result.append(json_data['result']['location']['lng'])
          tmp_result.append(cnt)
          print(cnt)
          result.append(tmp_result)
     file.close()
     return result


# print(content)
if __name__=='__main__':
     # get_pos()
     # m = folium.Map([33., 113.], tiles='stamentoner', zoom_start=5)
     m = folium.Map([33., 113.], zoom_start=5)
     data = get_pos()
     print(data)
     HeatMap(data).add_to(m)
     m.save('Heatmap.png')