# -*- coding: utf-8 -*-
"""
Iterates over all saved models and finds the 8x8 and 16x16 scores. Output
scores have the form:
    [chern_loss_8x8, dec_loss_8x8, chern_acc_8x8, dec_acc_8x8,
     chern_loss_16x16, dec_loss_16x16, chern_acc_16x16, dec_acc_16x16]
"""

import keras
from keras.models import Model, load_model
from keras.layers import Lambda, Input
from keras.optimizers import SGD
from keras import callbacks
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import pandas
import glob
import time
from custom_functions import *


# read data and labels from files - 1M test samples divided between 100 chunks
data_dir_8 = '/path/to/8x8_data'
input_size_8 = 8*8*3
data_chunks_8 = [s.replace('\\','/') for s in glob.glob(data_dir_8+'/*data*')]
labels_chunks_8 = [s.replace('\\','/') for s in glob.glob(data_dir_8+'/*labels*')]

chunk_list_8 = []
for chunk in range(len(data_chunks_8)):
    print(chunk)
    all_data = np.array(pandas.read_csv(data_chunks_8[chunk],header=None))
    all_data = all_data[:,0:all_data.shape[1]-1]
    all_labels = np.array(pandas.read_csv(labels_chunks_8[chunk],header=None))
    
    chunk_list_8.append(data_chunk(all_data, labels=all_labels, size=8, train_fraction=0))


# read data and labels from files - 1M test samples divided between 100 chunks
data_dir_16 = '/path/to/16x16_data'
input_size_16 = 16*16*3
data_chunks_16 = [s.replace('\\','/') for s in glob.glob(data_dir_16+'/*data*')]
labels_chunks_16 = [s.replace('\\','/') for s in glob.glob(data_dir_16+'/*labels*')]

chunk_list_16 = []
for chunk in range(len(data_chunks_16)):
    print(chunk)
    all_data = np.array(pandas.read_csv(data_chunks_16[chunk],header=None))
    all_data = all_data[:,0:all_data.shape[1]-1]
    all_labels = np.array(pandas.read_csv(labels_chunks_16[chunk],header=None))
    
    chunk_list_16.append(data_chunk(all_data, labels=all_labels, size=16, train_fraction=0))


# add custom losses to keras.losses to prevent an error when loading models
keras.losses.custom_loss = custom_loss
keras.losses.channel_loss = channel_loss
sgd = SGD(lr=0, decay=0, momentum=0.9, nesterov=True)


#%% Main Loop

score_table = np.zeros((100,8))

for i in range(100):

    model_8x8 = load_model('intermediate_models/8x8_model_partial_' + str(i) + '.h5')
    
    total_score = np.zeros((1,5))
    model_8x8.compile(loss=[custom_loss, channel_loss],
              loss_weights=[1.0,0.0],
              optimizer=sgd,
              metrics=['accuracy'])
    
    for chunk in range(len(data_chunks_8)):
        print(chunk)
        data = chunk_list_8[chunk]
        score = np.array(model_8x8.evaluate(data.data, [data.test_labels,data.test_avg_pooled], batch_size=640))
        print(score)
        total_score = total_score + score
        
    total_score = total_score / 100.0
    print(total_score[1:5])
    score_table[i,0:4] = total_score[0,1:5]


    
    # isolate RG layers
    model_RG = Model(inputs=model_8x8.input,
                     outputs=model_8x8.get_layer('final_4x4').output)

    # build 16x16 model
    inputs_16 = Input(shape=(input_size_16,), dtype='float32')
    x = model_RG(inputs_16)
    x = Lambda(unorder, name='unorder')(x)
    prediction_16 = model_8x8(x)
    model_16x16 = Model(inputs=inputs_16, outputs=prediction_16)
    
    total_score = np.zeros((1,5))
    model_16x16.compile(loss=[custom_loss, channel_loss],
              loss_weights=[1.0,0.0],
              optimizer=sgd,
              metrics=['accuracy'])
    
    for chunk in range(len(data_chunks_16)):
        print(chunk)
        data = chunk_list_16[chunk]
        score = np.array(model_16x16.evaluate(data.data, [data.test_labels,data.test_avg_pooled_2], batch_size=640))
        print(score)
        total_score = total_score + score
        
    total_score = total_score / 100.0
    print(total_score[1:5])
    score_table[i,4:8] = total_score[0,1:5]
    
numpy.savetxt("score_table.csv", score_table, delimiter=",")
