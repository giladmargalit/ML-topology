# -*- coding: utf-8 -*-
"""
Builds the "RG" model that reduces an 8x8 into a 4x4. Trains it initially to
act as an average pooling mapping (aka naive decimation).
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


# start measuring time
start_time = time.time()

# read data and labels from files
data_dir = '/path/to/unstructured/data'
input_size = 8*8*3
data_chunks = [s.replace('\\','/') for s in glob.glob(data_dir+'/*data*')]

#parameters for the reshaping and convolution
kernel_size = 3 #side length of square convolution kernel. Should be odd, >=3.
pad_size = int((kernel_size - 1) / 2) #padding needed to accommodate kernel
kernel_init = 'glorot_uniform'

#add custom losses to keras.losses to prevent an error when loading models
keras.losses.custom_loss = custom_loss
keras.losses.channel_loss = channel_loss


#%% Build and compile the model

# build the network
inputs = Input(shape=(input_size,), dtype='float32')
ordered_input = Lambda(order, name='order')(inputs)
x = ResBlock(ordered_input, filters=256, skip=0, stride=2, periodic=1)
for i in range(4):
    x = ResBlock(x, filters=512, skip=1, stride=1, periodic=1)
new_4x4 = Conv2D(3, 1, strides=(1,1), padding='valid',
                 data_format='channels_last', activation=None,
                 kernel_initializer=kernel_init, name='final_4x4')(x)
model = Model(inputs=inputs, outputs=new_4x4)


# compile model
sgd = SGD(lr=0.0005, decay=0, momentum=0.9, nesterov=True)
model.compile(loss=channel_loss,
              optimizer=sgd,
              metrics=['accuracy'])


#%% Train the model

'''
We iterate in chunks since the dataset is too large to be read as a whole. See
the data_chunk class in custom_functions
'''

acc_list = [] # accuracy per epoch
val_acc_list = [] # validation accuracy per epoch

# # load in chunks - comment out if already loaded (this takes several minutes)
chunk_list = []
for chunk in range(len(data_chunks)):
    print(chunk)
    all_data = np.array(pandas.read_csv(data_chunks[chunk],header=None))
    all_data = all_data[:,0:all_data.shape[1]-1]
    
    chunk_list.append(data_chunk(all_data, size=8))


for repeat in range(10):
    
    chunks = np.arange(0,len(chunk_list))
    np.random.shuffle(chunks)
    for chunk in chunks:
        
        #train on 8x8 lattices    
        train_data = chunk_list[chunk].train_data
        train_pooled = chunk_list[chunk].train_avg_pooled
    
        history = model.fit(train_data, train_pooled, batch_size=640,
                            callbacks=[callbacks.TerminateOnNaN()],
                            validation_split=0.2, epochs=1)
        
        acc_list[len(acc_list):] = history.history['loss']
        val_acc_list[len(val_acc_list):] = history.history['val_loss']
        


#%% plot results

plt.clf()
plt.plot(range(len(acc_list)), val_acc_list, color='blue', label='Val Loss')
plt.plot(range(len(acc_list)), acc_list, color='red', label='Train Loss')

print("--- RUNTIME: %.02f seconds ---" % (time.time() - start_time))

