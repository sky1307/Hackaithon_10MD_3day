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

dt = pd.read_csv('./data/test.csv')
print(dt.shape)

dr = pd.read_csv('./data/SonTay.csv')
print(dr.shape)

Q, H, Q_, H_ =[], [], [], []

for i in range(100): 
    dr.iloc[-5000:].to_csv('./data/SonTay.csv', index = False)
    model.data = model.generate_data()
    q_pred, h_pred = model.prediction_vp()
    Q.append(dt.iloc[i]['Q'])
    H.append(dt.iloc[i]['H'])
    Q_.append(q_pred)
    H_.append(h_pred)
    dr = dr.append(dt.iloc[i], ignore_index = True)
    print(i)
    

m, n = 0, 300
plt.subplots(1, figsize=(13, 7))
plt.plot(Q[m:n],marker=',', linewidth='0.8', label="Q" )
plt.plot(H[m:n],marker=',', linewidth='0.8', label="H" )
plt.plot(Q_[m:n],marker=',', linewidth='0.8', label="Q_" )
plt.plot(H_[m:n],marker=',', linewidth='0.8', label="H_" )
plt.xlabel('time')
plt.ylabel('Price')
plt.title('Test model')
plt.legend()
plt.show()
plt.savefig('./data/test')