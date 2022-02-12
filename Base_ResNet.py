# -*- coding: utf-8 -*-
"""
Builds and trains the base model for classifying 4x4 lattices by Chern number 
"""


import keras
from keras.models import Model
from keras.layers import Dense,Flatten,MaxPooling2D,Lambda
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
data_dir = 'path/to/data'
input_size = 4*4*3
data_chunks = [s.replace('\\','/') for s in glob.glob(data_dir+'/*data*')]
labels_chunks = [s.replace('\\','/') for s in glob.glob(data_dir+'/*labels*')]


#%% Build and compile the model

# build the network
inputs = Input(shape=(input_size,), dtype='float32')
ordered_input = Lambda(order, name='order')(inputs)
x = ResBlock(ordered_input, filters=256, layer_num=1, skip=0)
for i in range(4):
    x = ResBlock(x, filters=256, layer_num=2, skip=1)
x = MaxPooling2D(pool_size=(4,4))(x)
x = Flatten()(x)
prediction = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=prediction)

# compile
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
model.compile(loss=custom_loss,
              optimizer=sgd,
              metrics=['accuracy'])

#%% train the network and get the score

pat = 50 # patience for callbacks to determine how often to reduce lr
Reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=pat, mode='auto', verbose=1, min_lr=1e-6)
EarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=(pat + 1))

acc_list = []
val_acc_list = []

# # load in chunks - comment out if already loaded (this takes several minutes)
chunk_list = []
for chunk in range(len(data_chunks)):
    print(chunk)
    all_data = np.array(pandas.read_csv(data_chunks[chunk],header=None))
    all_data = all_data[:,0:all_data.shape[1]-1]
    all_labels = np.array(pandas.read_csv(labels_chunks[chunk],header=None))
    
    chunk_list.append(data_chunk(all_data, labels=all_labels))
    
# iterate over chunks of the dataset
# (because the dataset is too large to be read as a whole)
for repeat in range(100):
    
    chunks = np.arange(0,len(chunk_list))
    np.random.shuffle(chunks)
    for chunk in chunks:
        
        train_data = chunk_list[chunk].train_data
        train_labels = chunk_list[chunk].train_labels
        test_data = chunk_list[chunk].test_data
        test_labels = chunk_list[chunk].test_labels
        
        # train
        history = model.fit(train_data, train_labels, batch_size=640,
                            callbacks=[callbacks.TerminateOnNaN(), Reduce, EarlyStop],
                            validation_split=0.2, epochs=1)
       
        # # uncomment if testing is necessary once per loop
        # if chunk == len(data_chunks)-1:
        #     score,accuracy = model.evaluate(test_data, test_labels, batch_size=16, verbose=0)
        #     print("Accuracy = %.04f" % accuracy)
        
        acc_list[len(acc_list):] = history.history['acc']
        val_acc_list[len(val_acc_list):] = history.history['val_acc']
    
    
#%% plot results

plt.clf()
plt.plot(range(len(acc_list)), acc_list, color='red', label='Train acc')
plt.plot(range(len(acc_list)), val_acc_list, color='blue', label='Val acc')


print("--- RUNTIME: %.02f seconds ---" % (time.time() - start_time))

