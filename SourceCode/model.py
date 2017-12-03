# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:20:08 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Fri Nov 10 19:20:08 2017
"""

import util
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D,MaxPooling2D
from collections import deque

class Research_Models():
    def __init__(self,model_name,seq_length,saved_model=None,feature_dim=2048,no_cls):
        """  This class contains two models one with LSTM only and other with LSTM+CNN
        Later it will be updated with LSTM+Inception and CNN only(Inception V4).
        Default is Pre-trained Inception with LSTM
        """
        self.seq_length=seq_length
        self.saved_model=saved_model
        self.no_cls=no_cls
        self.features_dim=features_dim
        self.feature_queue = deque()
        # Setting metrics for the compliation of models in keras
        metrics=['accuracy']
        if self.no_cls>=20:
            # Default is k=5
            metrics.append('top_k_categorical_accuracy')
        
        if self.saved_model is not None:
             print("Loading model %s" % self.saved_model)
             self.model = load_model(self.saved_model)
        elif model_name=='CNN_LSTM':
            print("Loading CNN_LSTM model.")
            self.input_shape = (self.seq_length, self.features_dim[0],self.features_dim[1],self.features_dim[2])
            self.model=self.cnn_lstm()
        else:
            print("Loading LSTM model.")
            self.input_shape = (self.seq_length, self.features_dim)
            self.model=self.lstm()
        
    def lstm(self):
        "This function is a simple LSTM model which takes output of the Inception model as input and returns the softmax for each class"
        model=Sequential()
        model.add(LSTM(self.feature_dim,return_sequences=False,input_shape=self.input_shape,dropout=0.5))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.no_cls,activation='softmax'))
        return model
    def cnn_lstm(self):
        """  Building a CNN with LSTM (for full training). The CNN model used is based on a model previously developed for a course work problem"""
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2),activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3,3),padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(256, (3,3),padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(512, (3,3),padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model