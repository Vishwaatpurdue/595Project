# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:59:07 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Sun Nov 12 01:59:07 2017
"""

import util
import csv
import glob
import threading
class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class Dataset():
    def __init__(self,cls_lmt=None,img_shape=(240,240,3),data_length=40,maxframes=300,path=None):
        """ data_length= no.of frames to be considered
            class_limit= to limit the data to certain classes options (None - no limit)
        """
        self.data_length=data_length
        #self.cls_lmt=cls_lmt
        self.maxframes=maxframes # max video length is 10 seconds  is considered      
        
        self.datafile=self.fetch_datafile()
        # Get the sorted classes till the limit or all the classes
        self.cls=self.get_classes()
        self.img_shape=img_shape
        self.seq_path=path
        return
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
        return sorted(cls)
    def one_hot_encoding(self,cls_name):
        """ Given a class name it provides the corresponding target vector """
        target_idx=self.cls.index(cls_name)
        target_hot=np.zeros(len(self.cls),dtype=int)
        target_hot[target_idx]=1
        return target_hot
    def split_dataset(self):
        """  Splitting the datase into train validate and test datasets. The split between training and validate is 80:20"""
        train=[]
        validate=[]
        test=[]
        for item in self.datafile:
            if item[0]=='traindataset':
                train.append(item)
            else:
                test.append(item)
        max_train=len(train)-1
        valid_split=int(max_train*0.2)
        random_gen=sorted(np.random.randint(0,max_train,valid_split))
        temp=np.asanyarray(train)
        validate=(temp[random_gen]).tolist()
        (random_gen.tolist()).reverse()
        for i in random_gen:
            train.remove(train[i])
        return train,test,validate
    def load_data(self,data_cat,data_type):
        """  This functions loads the data into the memory so as to train faster"""
        train,test,validate=self.split_dataset()
        if data_cat=='train':        
            data=train
        elif data_cat=='validate':
            data=validate
        else:
            data=test
        print("Loading %d samples into the memory for %sing..."%(len(data),data_cat))
        Ip,target=[],[]
        for row in data:
            if data_type=='images':
                frames=self.get_frames(row)
                frames=self.rescale_frames(frames)
                seq=self.build_sequence(frames)
            else:
                seq=self.get_sequences(data_type,row)
                if seq==None:
                    raise ValueError("Sequence not found")
            Ip.append(seq)
            target.append(self.one_hot_encoding(row[1]))
        return np.asanyarray(Ip),np.asanyarray(target)
    def get_frames(self,row):
        """ Given a row of data file. It loads the images of the video described in the datafile """
        path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data',row[0],row[1],row[2])
        # if not sorted it loads the images in any order which is not useful in learning the video
        imgs=sorted(glob.glob(path+'*jpg'))
        return imgs
    def rescale_frames(self,frames):
        """ Given a list of frames(imgs/features) select only the required images to understand about the video. 
        Example: if seq length is 30 and the total images is 300 every one in ten images is taken.
        So that it can understand the video."""
        try:
            # Check whether the video length is > than sequence length
            assert len(frames)>=self.data_length
            # Find the no.of frames to be skipped
            skip_no=len(frames)//self.data_length
            # Extract the required frames
            req_frames=[frames[i] for i in range(0,len(frames),skip_no)]
            # remove the extra frames than the required length
            return req_frames[:self.data_length]
        except AssertionError:
            print("Clean the data. Keep only video files which are larger than sequence length.")
            raise
            return
        
    def build_sequence(self,frames):
        """ This function loads the image, reshapes it and normalizes it in range of (0,1). This method also normalizes to (-1,+1)"""
        op_frames=[]
        for frame in frames:
            img=cv2.imread(frame)
            img=cv2.resize(img,self.img_shape[0:1],cv2.INTER_CUBIC)
            #frame=(frame-128)/128
            #img=img[:,:,::-1]
            img=img/255.0
            op_frames.append(img)
        return
    def get_sequences(self,data_type,row):
        """ Get the saved sequences of features/images. The sequences are saved in numpy format as .npy"""
        path=os.path.join(self.seq_path,row[2]+'-'+str(self.data_length)+'-'+data_type+'.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            print("File not found. So skipping..")
        return
    @threadsafe_generator
    def frame_generator(self,batch_size,data_cat,data_type):
        """ This is a frame generator which is similar to load data into the memory. 
        It is useful in case of parallel processing either using multiple workers of the CPU or a GPU.
        When loading data to memory is expensive use this approach.
        It can be used for both images and features"""
        train,test,validate=self.split_dataset()
        if data_cat=='train':        
            data=train
        elif data_cat=='validate':
            data=validate
        else:
            data=test
        print("Loading %d samples into the memory for %sing..."%(len(data),data_cat))
        while True:
            Ip,target=[],[]
            # Batch-wise sequences
            for i in range(batch_size):
                seq=None
                row_idx=np.random.randint(0,len(data)-1)
                row=data[row_idx]
                if data_type=='images':
                    frames=self.get_frames(row)
                    frames=self.rescale_frames(frames)
                    seq=self.build_sequence(frames)
                else:
                    seq=self.get_sequences(data_type,row)
                    if seq==None:
                        raise ValueError("Sequence not found")
                Ip.append(seq)
                target.append(self.one_hot_encoding(row[1]))
            yield np.asanyarray(Ip),np.asanyarray(target)
        return
    
    