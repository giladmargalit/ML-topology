# -*- coding: utf-8 -*-
"""
Loads the RG model that was initialized in RG_ResNet_initialize_to_decimation.py,
then trains it to classify by Chern number. Optionally, a 2nd loss can be used
that rewards the model for remaining close to average pooling (we found best
results when the weight for this 2nd loss was set to 0.

This code saves a model after each epoch (after all samples are processed once),
which can then be analyzed on both 8x8 and 16x16 lattices with get_scores.py.
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
data_dir = '/path/to/8x8_data'
input_size = 8*8*3
data_chunks = [s.replace('\\','/') for s in glob.glob(data_dir+'/*data*')]
labels_chunks = [s.replace('\\','/') for s in glob.glob(data_dir+'/*labels*')]

# add custom losses to keras.losses to prevent an error when loading models
keras.losses.custom_loss = custom_loss
keras.losses.channel_loss = channel_loss


#%% build and compile the model

# load 4x4 model and freeze weights
model_4x4 = load_model('4x4_model.h5')
model_4x4.name = 'chern'
for layer in model_4x4.layers:
    layer.trainable = False

# load 8x8 model already trained to replicate naive decimation
# (from RG_ResNet_initialize_to_decimation.py)
decimation_model = load_model('8x8_decimation_model.h5')
decimation_model.name = "dec"

# build the network
inputs = Input(shape=(input_size,), dtype='float32')
ordered_4x4 = decimation_model(inputs) # reduce to 4x4
unordered_4x4 = Lambda(unorder, name='unorder')(ordered_4x4) # base model takes in unordered data
prediction = model_4x4(unordered_4x4) # feed computed 4x4 into loaded 4x4 model
model = Model(inputs=inputs, outputs=[prediction, ordered_4x4])

# set loss weights
chern_weight = K.variable(1.0) # how much the custom loss acting on the chern number should contribute to total loss
dec_weight = K.variable(0.0) # how much the similarity to the decimation map should contribute to total loss

# compile model
sgd = SGD(lr=0.0005, decay=0, momentum=0.9, nesterov=True)
model.compile(loss=[custom_loss, channel_loss],
              loss_weights=[chern_weight,dec_weight],
              optimizer=sgd,
              metrics=['accuracy'])


#%% train the model

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
    all_labels = np.array(pandas.read_csv(labels_chunks[chunk],header=None))
    
    chunk_list.append(data_chunk(all_data, labels=all_labels, size=8))

count = 0

for repeat in range(100):
    
    count = count + 1
    chunks = np.arange(0,len(chunk_list))
    np.random.shuffle(chunks)
    
    for chunk in chunks:
        
        train_data = chunk_list[chunk].train_data
        train_labels = chunk_list[chunk].train_labels
        train_pooled = chunk_list[chunk].train_avg_pooled
        
        test_data = chunk_list[chunk].test_data
        test_labels = chunk_list[chunk].test_labels
        test_pooled = chunk_list[chunk].test_avg_pooled

        history = model.fit(train_data, [train_labels,train_pooled], batch_size=640,
                            callbacks=[callbacks.TerminateOnNaN()],
                            validation_split=0.2, epochs=1)
        
        # # uncomment if testing is necessary once per loop
        # if chunk == len(data_chunks)-1:
        #     score = model.evaluate(test_data, [test_labels,test_pooled], batch_size=640)
        #     print("Accuracy = %.04f" % score[4])

        acc_list[len(acc_list):] = history.history['chern_acc']
        val_acc_list[len(val_acc_list):] = history.history['val_chern_acc']


    model_str = 'intermediate_models/8x8_model_partial_' + str(count) + '.h5'
    try:
        model.save(model_str)
    except:
        pass


#%% plot results

plt.clf()
plt.plot(range(len(acc_list)), val_acc_list, color='blue', label='Val acc')
plt.plot(range(len(acc_list)), acc_list, color='red', label='Train acc')

print("--- RUNTIME: %.02f seconds ---" % (time.time() - start_time))

