# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:30:13 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Mon Dec  4 20:30:13 2017
"""

import cv2
import os
import numpy as np
from keras.models import load_model
from model import Research_Models 
from extract_features import extract_features

"""
prompt="Enter the absolute path for the video to be analyzed"
path=input(prompt)
prompt="Enter the sequence length"    
seq_length=np.int64(input(prompt))
if seq_length == '':
    seq_length=None
prompt="Enter the maximum frame length"    
max_frames=np.int64(input(prompt))
if max_frames == '':
    max_frames=None
    raise ValueError("Enter Valid interger for max frames")
prompt="Enter the model_name from the list \n 1.LSTM \n 2.CNN_LSTM"    
model_name=input(prompt)
if model_name == '':
    model_name=None
if model_name=='LSTM':
    data_type='features'
    img_shape=None
    # Feature Extraction
    extract_features(seq_length,max_frames,abspath=None)
elif model_name=='CNN_LSTM':
    data_type='images'
    img_shape=(100,100,3)
else:
    raise ValueError("Invalid Model Selected. Select from List given in the options")
"""    
path="/home/vishwa/595/595Project/data/testdataset/Rowing/v_Rowing_g01_c02.avi"
seq_length=40
max_frames=300
model_name='LSTM'
seq_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data', 'sequences')
cls_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','traindataset')
cls=sorted(os.listdir(cls_path))

# extract the features
if model_name == '':
    model_name=None
if model_name=='LSTM':
    data_type='features'
    img_shape=None
    # Feature Extraction
    extract_features(seq_length,max_frames,abspath=None)
elif model_name=='CNN_LSTM':
    data_type='images'
    img_shape=(100,100,3)
else:
    raise ValueError("Invalid Model Selected. Select from List given in the options")
# load the data
X=np.load(seq_path)
# loading the model
saved_model=input("Enter the absolute path for the model to be used")
rm=Research_Models(model_name=model_name,seq_length=seq_length, saved_model=saved_model,feature_dim=img_shape,no_cls=101)
# Classify the image
pred=rm.model.predict(X)
pred=np.argmax(pred)
# Extract the predicted class name
cap=cv2.VideoCapture(path)
frame_wd=int(cap.get(3))
frame_ht=int(cap.get(4))
fname=os.path.join(os.path.dirname(os.path.dirname(__file__)),'Output',path.split('/')[-1])
FOUR_CC=cv2.VideoWriter_fourcc('M','J','P','G')
fname=fname
out = cv2.VideoWriter(fname,FOUR_CC, 30, (frame_wd,frame_ht),1)
while(True):
    ret,frames=cap.read()    
    font = cv2.FONT_HERSHEY_SIMPLEX    
    if ret==True:
        cv2.putText(frames, cls[pred], (100, 50), font, 0.8, (0, 255, 0))
        out.write(frames)
        cv2.imshow('op',frames)
        cv2.waitKey(33)
        cv2.destroyWindow('op')
    else:
        cv2.destroyWindow('op')
        break
cap.release()
out.release()
cv2.destroyAllWindows();