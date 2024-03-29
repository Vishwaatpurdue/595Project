# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:19:28 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Fri Nov 10 19:19:28 2017

This script generates the features that are used by the training and testing script to train the LSTM network. 
This script has to executed before executing the training script. User Inputs are sequence length, img_dim to get results 
that could be used to analyse the algorithm.
"""
import cv2
import util
from dataloader import Dataset
from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.applications.inception_v4 import InceptionV4, preprocess_input as preproc
from keras.models import Model, load_model
from keras.layers import Input 
class Feature_Extractor():
    def __init__(self,weights=None,layers2pop=None):
        """ This class is developed to extract features using InceptionV3 model. 
        If weights are not provided. It loads the defaults weights pre-trained on the ILSRV dataset(Imagenet). 
        This class uses the keras implementation of Inception models."""
        self.weights=weights
        if self.weights==None:
            base_model=InceptionV3(weights='imagenet',include_top=True)
            # Extracting features in the last layer before the classifier layer
            self.model=Model(inputs=base_model,output=base_model.get_layer('avg_pool').output)
        else:
            self.model=load_model(self.weights)
            # Removing the softmax layer and Other layers depending on user input
            for i in layers2pop:
                self.model.layers.pop()
            self.model.outputs=[self.model.layers[-(layers2pop-1)].output]
            self.model.output_layers=[self.model.layers[-(layers2pop-1)]]
            self.model.layers[-(layers2pop-1)].outbound_nodes=[]
        return
    def extract(self,path,target_size):
        img = cv2.imread(path)
        img=cv2.resize(img,target_size,cv2.INTER_CUBIC)
        x=preprocess_input(img)
        features=self.model.predict(x)
        features=features[0]        
        return features


def main():
    prompt="Enter the sequence length and max frames to be considered for feature extraction with spaces..."
    usr_data=raw_input(prompt)
    usr_data=usr_data.split(' ')
    ip=[np.int64(i) for i in usr_data]
    seq_length,max_frames=ip
    prompt="Enter the absolute file path where data sequences need to be created(sequences folder will be created if not present...) \n If default press enter"
    abspath=raw_input(prompt)
    if abspath=='':
        seq_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data', 'sequences')
    else:        
        if os.path.isdir(os.path.join(abspath,'sequences')):
            pass
        else:
            os.mkdir(os.path.join(abspath,'sequences'))
    data = Dataset(data_length=seq_length,maxframes=max_frames,path=seq_path)
    feature_model=Feature_Extractor()
    for video in data.datafile():
        path = os.path.join('data','sequences',video[2]+'-'+str(seq_length)+'-features')
        if os.path.isfile(path + '.npy'):
            continue
        frames=data.get_frames(video)
        # Skip intermiediate frames
        frames=data.rescale_frames(frames)
        seq=[]
        for frame in frames:
            features=feature_model.extract(frame)
            seq.append(features)
        np.save(path,seq)
        
def extract_features(seq_length,max_frames,abspath=None):
    """ This function is to ease the training process to analyze the performance of the algorithm for various hyper parameters."""
    if abspath=='' or abspath==None:
        seq_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data', 'sequences')
    else:        
        if os.path.isdir(os.path.join(abspath,'sequences')):
            continue
        else:
            os.mkdir(os.path.join(abspath,'sequences'))
    data = Dataset(data_length=seq_length,maxframes=max_frames,path=seq_path)
    feature_model=Feature_Extractor()
    for video in data.datafile():
        path = os.path.join('data','sequences',video[2]+'-'+str(seq_length)+'-features')
        if os.path.isfile(path + '.npy'):
            continue
        frames=data.get_frames(video)
        # Skip intermiediate frames
        frames=data.rescale_frames(frames)
        seq=[]
        for frame in frames:
            features=feature_model.extract(frame)
            seq.append(features)
        np.save(path,seq)
    
if __name__=='__main__':   
    main()     
        
    
    
    
