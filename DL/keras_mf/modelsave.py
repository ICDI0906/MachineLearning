# @Time : 2018/10/11 下午7:30 
# @Author : Kaishun Zhang 
# @File : modelsave.py 
# @Function: 模型保存和提取

from keras.models import load_model
from keras.models import Model
model = Model()
model.save('mymodel.h5')  # need to install h5py
model = load_model('mymodel.h5') #load the model from h4 file