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
model.retransform_prediction_vp(mode='roll')
# model.evaluate_model(mode='roll')