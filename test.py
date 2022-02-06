import sys
from tensorflow import keras
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ensemble import Ensemble
import tensorflow.keras.backend as K

with open("./settings/model/config.yaml", 'r') as stream:
        config =yaml.load(stream ,Loader=  yaml.FullLoader)

K.clear_session()
sigma_index_lst = [0,1]
epoch_num = 8
model = Ensemble(mode='test', model_kind='rnn_cnn', sigma_lst=sigma_index_lst,
                     default_n=20, epoch_num=epoch_num, epoch_min=100, epoch_step=50, **config)
model.train_model_outer()
x_ = np.array([[[35640.21, 35586.79],
        [35662.31, 35610.81],
        [35676.21, 35644.61],
        [35711.48, 35676.21],
        [35710.73, 35675.96],
        [35756.58, 35701.48],
        [35800.95, 35702.99],
        [35793.67, 35693.71],
        [35810.26, 35759.09],
        [35877.83, 35756.59],
        [35872.95, 35817.99],
        [35825.08, 35783.42],
        [35817.46, 35774.11],
        [35824.07, 35794.9 ],
        [35823.1 , 35800.01],
        [35881.59, 35818.66],
        [35881.59, 35820.33],
        [35865.58, 35829.51],
        [35897.  , 35860.37],
        [35934.  , 35879.06],
        [35920.5 , 35866.14],
        [35895.35, 35827.81],
        [35900.  , 35835.03],
        [35902.87, 35848.82],
        [35876.  , 35820.02],
        [35827.  , 35800.95],
        [35824.58, 35802.22],
        [35854.69, 35806.67],
        [35870.01, 35836.79],
        [35906.3 , 35868.09]]])

# print(x_)
print(type(x_))
model.prediction_vp(x_)
# # model.evaluate_model(mode='roll')

# data_file = './data/SonTay.csv'
# df = pd.read_csv (data_file).iloc[:120]

# def extract_window_data(df, window_len=30):
#     window_data = []
#     for idx in range(len(df) - window_len):
#         tmp = df[idx: (idx + window_len)].copy()
#         window_data.append(tmp.values)
#     return np.array(window_data)

# def prepare_data(df, window_len=30):
#     data_x = extract_window_data(df, window_len)
#     data_y = df[window_len:].values
#     return data_x, data_y
# data_x, data_y = prepare_data(df, window_len=30)


# y_pred = []
# m =  None
# for x in data_x:
#     x_ = x[:,2:4][np.newaxis, :]
#     print(x_)
#     print(type(x))
#     m = x_
#     break

# y_ = model.prediction_vp(m)
# y_pred.append(y_)