# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:59:07 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Sun Nov 12 01:59:07 2017
"""

import util
import csv
class Dataset():
    def __init__(self,cls_lmt=None,img_shape=(240,240,3)):
        """ data_length= no.of frames to be considered
            class_limit= to limit the data to certain classes options (None - no limit)
        """
        self.data_length=50
        #self.cls_lmt=cls_lmt
        self.maxframes=30*10 # max video length is 10 seconds        
        
        self.datafile=self.fetch_datafile()
        # Get the sorted classes till the limit or all the classes
        self.cls=self.get_classes()
        self.img_shape=img_shape
    
    def fetch_datafile(self):
        """ Load data from the folders """
        fpath=os.path.dirname(os.path.dirname(__file__))+"/data/Datainfo.csv"
        with open(fpath,'r') as f:
            fread=csv.reader(f)
            data=list(fread)
        return data
    def get_classes(self):
        """ Load classes from the folder names """
        cls=[]
        for val in self.data:
            if val[1] not in cls:
                cls.append(val[1])
        return cls
    def one_hot_encoding(self):
        