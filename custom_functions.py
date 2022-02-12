# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:52:56 2019

@author: giladm
"""


import keras
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense,Activation,Lambda,Input,Conv2D,AveragePooling2D,BatchNormalization,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras import regularizers
from keras import callbacks
import numpy as np
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
import glob
import time
import os
from datetime import datetime
from timeit import default_timer as timer



def order(x, ch=3):
    import tensorflow as tf
    import numpy as np
    
    '''
    Converts input tensor of shape (?, 3N^2) into lattice of shape (?, N, N, 3)
    '''
    
    dim = int(np.sqrt(int(x.get_shape()[1]) / ch))
    x = tf.reshape(x,[-1,ch,dim,dim])
    x = tf.transpose(x,[0,2,3,1])
    return x


def unorder(x, ch=3):
    import tensorflow as tf
    import numpy as np
    
    '''
    inverse of order
    '''
    
    dim = int(x.get_shape()[1])
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x,[-1,dim*dim*ch])
    #print(x.shape)
    return x


def order_array(x, ch=3):
    import numpy as np
    
    '''
    Converts input array of shape (?, 3N^2) into lattice of shape (?, N, N, 3)
    '''
    
    dim = int(np.sqrt(int(x.shape[1]) / ch))
    x = np.reshape(x,(-1,ch,dim,dim))
    x = np.transpose(x,(0,2,3,1))
    return x


def unorder_array(x, ch=3):
    import numpy as np
    
    '''
    inverse of order_array
    '''
    
    dim = int(x.shape[1])
    x = np.transpose(x,(0,3,1,2))
    x = np.reshape(x,(-1,ch*dim*dim))
    return x


def periodic_padding(image, padding=2):
    import tensorflow as tf
    import numpy as np
    
    '''
    Creates a periodic padding around the image so that 'valid' padding in the
    2D convolution layer respects the boundary conditions and maintains size
    '''

    upper_pad = image[:, -padding:, :, :]
    lower_pad = image[:, :padding, :, :]

    partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)

    left_pad = partial_image[:, :, -padding:, :]
    right_pad = partial_image[:, :, :padding, :]

    padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)

    return padded_image

    
def custom_loss(y_true,y_pred):
    import tensorflow as tf
    from keras import backend as K
    
    '''
    Our custom loss function, |x| + x^2. Responds strongly to deviations far
    from 0 without becoming too weak for values close to 0. This loss is used
    to match Chern number results to labels.
    '''
    
    return K.mean(K.mean(K.abs(y_true - y_pred) + K.square(y_true - y_pred),axis=-1),axis=-1)


def channel_loss(y_true,y_pred):
    import tensorflow as tf
    import numpy as np
    from keras import backend as K
    
    '''
    The loss used to match output 4x4 lattices to the 8x8 inputs reduced by
    AveragePool2D. It acts as our custom_loss loss function on every site in
    every channel, weights each channel by channel_weight_vec, then sums. We
    achieved good results using equal weights (despite Delta being an order of
    magnitude smaller than the other channels) but perhaps different weights
    would improve accuracy further.
    '''

    channel_weight_vec = tf.convert_to_tensor([1,1,1.0])
    diff = K.sum(channel_weight_vec * K.mean(K.mean(K.abs(y_true - y_pred) + K.square(y_true - y_pred),axis=-2),axis=-2),axis=-1)
    return diff


def ResBlock(x, filters, kernel_size=3, reg=[0,0,0], drop=0, layer_num=2, skip=1, periodic=0, stride=1):
    import tensorflow as tf
    import numpy as np
    from keras import backend as K
    from keras.layers import Activation,Dropout,Conv2D,BatchNormalization,Lambda,Add
    
    '''
    Defines a convolutional block with a ResNet skip connection between the
    first and last layer.
     - x: layer input
     - filters: number of kernels in each layer; if stride=2, all layers after
     the first use 2*filters instead, in order to keep the number of parameters
     constant for each layer in the block.
     - kernel_size: kernel size
     - reg: [kernel_regularizer, bias_regularizer, activation_regularizer].
     Regularization is the same in all layers.
     - drop: dropout rate; identical in all layers
     - layer_num: number of convolutional layers in the block
     - skip: flag to set skip connection. If 0, the skip is not used.
     - periodic: periodic padding flag. If 0, use 'same' padding (zero-pad).
     If 1, use 'valid' padding with a periodic padding step (periodic-pad).
     - stride: stride of the first layer; if stride=2, the block will lead to
     overall halving of lattice size along each dimension.
    '''
    
    #get parameters
    reg_kernel, reg_bias, reg_act = reg
    pad_size = int((kernel_size - 1) / 2) #padding needed to accommodate kernel
    kernel_init = keras.initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='normal', seed=None)
    
    if periodic:
        padding_type = 'valid' #don't use padding, since the Lambda layer adds the pad beforehand
    else:
        padding_type = 'same' #pad with zeros

    # store the shortcut array and, if stride=2, implement the Conv2D layer
    # that reduces the lattice
    if (stride > 1):
        if periodic:
            x = Lambda(periodic_padding, arguments={'padding': pad_size})(x)
        x2 = Conv2D(filters*stride, 1, strides=(stride,stride),
                    padding=padding_type, data_format='channels_last',
                    activation=None, kernel_initializer=kernel_init)(x)
        x = Conv2D(filters, kernel_size, strides=(stride,stride),
                   padding=padding_type,
                   data_format='channels_last', activation='relu',
                   kernel_regularizer=regularizers.l2(reg_kernel),
                   bias_regularizer=regularizers.l2(reg_bias),
                   activity_regularizer=regularizers.l2(reg_act),
                   kernel_initializer=kernel_init)(x)
        if (drop > 0):
            x = Dropout(drop)(x)
        layer_num = layer_num - 1
    else:
        x2 = x
    
    for i in range(layer_num):
        if periodic:
            x = Lambda(periodic_padding, arguments={'padding': pad_size})(x)
        x = Conv2D(filters*stride, kernel_size, padding=padding_type,
                   data_format='channels_last', activation='relu',
                   kernel_regularizer=regularizers.l2(reg_kernel),
                   bias_regularizer=regularizers.l2(reg_bias),
                   activity_regularizer=regularizers.l2(reg_act),
                   kernel_initializer=kernel_init)(x)
        if (drop > 0):
            x = Dropout(drop)(x)
    
    if skip:
        # add the shortcut
        x = Add()([x,x2])
        x = Activation('relu')(x)
    
    return x


class data_chunk():
    
    '''
    Our datasets are divided between 100 chunks since they are too large to
    read at once. For each chunk, we define the training and testing sets, then
    additionally (if 8x8 or 16x16) find the 4x4 lattice that corresponds to the
    samples after average pooling reduces them to size 4x4.
    '''
    
    def __init__(self, data, labels=None, size=4, train_fraction=0.8):
        self.data = data
        self.labels = labels

        test_fraction = 1 - train_fraction
        
        # calculate the number of samples used for training and for testing
        num_samples = data.shape[0]
        num_train = round(train_fraction * num_samples)
        num_test = round(test_fraction * num_samples)
        
        # set the training and testing arrays - data and labels
        self.train_data = data[0:num_train,:]
        self.test_data = data[num_train:(num_train+num_test),:]
        
        if (labels is not None):
            self.train_labels = labels[0:num_train,:]
            self.test_labels = labels[num_train:(num_train+num_test),:]
        
        # compute the effective 4x4 after average pooling
        self.train_ord = order_array(self.train_data)
        self.test_ord = order_array(self.test_data)
        
        if (size > 4):
            self.train_avg_pooled = (self.train_ord[:,::2,::2,:] + self.train_ord[:,1::2,::2,:] \
                                    + self.train_ord[:,::2,1::2,:] + self.train_ord[:,1::2,1::2,:]) / 4
            self.test_avg_pooled = (self.test_ord[:,::2,::2,:] + self.test_ord[:,1::2,::2,:] \
                                    + self.test_ord[:,::2,1::2,:] + self.test_ord[:,1::2,1::2,:]) / 4

        if (size > 8):
            self.train_avg_pooled_2 = (self.train_avg_pooled[:,::2,::2,:] + self.train_avg_pooled[:,1::2,::2,:] \
                                    + self.train_avg_pooled[:,::2,1::2,:] + self.train_avg_pooled[:,1::2,1::2,:]) / 4
            self.test_avg_pooled_2 = (self.test_avg_pooled[:,::2,::2,:] + self.test_avg_pooled[:,1::2,::2,:] \
                                    + self.test_avg_pooled[:,::2,1::2,:] + self.test_avg_pooled[:,1::2,1::2,:]) / 4

                
def tics(stamp=False):
    if stamp:
        print(datetime.now().strftime("Starting at %d/%m/%Y %H:%M:%S"))
    return timer()

    
def toc(tic, disp=True, nice_format=False, stamp=False):
    elapsed_time = timer() - tic
    if stamp:
        print(datetime.now().strftime("Finished at %d/%m/%Y %H:%M:%S"))
    if disp:
        if (not nice_format) or (elapsed_time < 60):
            print("Elapsed time is " + str(np.round(elapsed_time,3)) + " seconds.")
        else:
            hrs = elapsed_time // 3600
            mins = elapsed_time % 3600 // 60
            secs = elapsed_time % 3600 % 60
            time_str = '{:02d}:{:02d}:{:02d}'.format(int(hrs), int(mins), int(secs))
            if hrs > 0:
                type_str = "hours"
            else:
                type_str = "minutes"
                time_str = time_str[3:]
            print("Elapsed time is " + time_str + " " + type_str + ".")
    return elapsed_time
