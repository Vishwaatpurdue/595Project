# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:20:30 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Fri Nov 10 19:20:30 2017
"""

import util
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from model import Research_Models
from dataloader import Dataset
from extract_features import extract_features

def train(seq_length,model_name,saved_model,data_type,cls_lmt=None,img_shape=None,batch_size=32,no_epoch=100):
    
    
    # creating setting for the training and saving the best model after training
    tb=TensorBoard(log_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data', 'logs', model_name))
    early_stop= EarlyStopping(patience=10)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','logs',model+'-'+'training,-'+str(timestamp)+ '.log'))
    checkpointer=ModelCheckpoint(fpath=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data', 'checkpoints', model + '-' + data_type +'.{epoch:04d}-{val_loss:.4f}.hdf5'),\
    verbose=1,save_best_only=True)
        
    #Loading Inputs
    if image_shape is None:
        data = DataSet(data_length=seq_length,cls_lmt=cls_lmt)
    else:
        data = DataSet(data_length=seq_length,cls_lmt=cls_lmt,image_shape=image_shape)
        
    # Compute no of steps per epoch as tensorflow uses only iteration not epochs
    steps_per_epoch=(len(data.data) * 0.7) // batch_size
    
    # load the dataset for training and validation
    X, y = data.load_data('train', data_type)
    X_val, y_val = data.load_data('validate', data_type)
    
    # Load the model
    rm = Research_Models(model_name=model_name, seq_length=seq_length, saved_model=saved_model,feature_dim=img_shape,no_cls=len(data.cls))
    rm.model.fit(X,y,batch_size=batch_size,validation_data=(X_val, y_val),verbose=1,callbacks=[tb, early_stop, csv_logger],epochs=no_epoch)
    return 

def test(model_name,saved_model,data_type,seq_length,cls_lmt=None,img_shape=None,batch_size=30):
    #Loading Inputs
    if image_shape is None:
        data = DataSet(data_length=seq_length,cls_lmt=cls_lmt)
    else:
        data = DataSet(data_length=seq_length,cls_lmt=cls_lmt,image_shape=image_shape)
    X, y = data.load_data('test', data_type)
    rm = Research_Models(model_name=model_name,seq_length=seq_length, saved_model=saved_model,feature_dim=img_shape,no_cls=len(data.cls))
    pred=rm.model.predict(X,verbose=0)
    corr=0
    for i in range(len(y)):
        if y[i]==pred[i]:
            corr+=1
    tot=len(y)
    acc=corr/tot*100
    print("Test Accuracy is:%0.2f"%(acc))
    
    return acc
    
def main():
    """ Training and test settings for the algorithm."""
    model_name='lstm'
    saved_model=None
    cls_lmt=None
    batch_size=32
    prompt="Enter the sequence length"    
    seq_length=np.int64(raw_input(prompt))
    if seq_length == '':
        seq_length=None
    prompt="Enter the maximum frame length"    
    max_frames=np.int64(raw_input(prompt))
    if max_frames == '':
        max_frames=None
        raise ValueError("Enter Valid interger for max frames")
    prompt="Enter the model_name from the list \n 1.LSTM \n 2.CNN_LSTM"    
    model_name=np.int64(raw_input(prompt))
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
    
    # Training the algorithm
    train(seq_length,model_name,saved_model,data_type,cls_lmt=None,img_shape=img_shape,batch_size=32,no_epoch=100)
    # Testing the algorithm
    saved_model=raw_input("Enter the absolute path for the model saved after training")
    acc=test(model_name,saved_model,data_type,seq_length,cls_lmt=None,img_shape=None,batch_size=30)
    with open('Testfile.csv','a+') as f:
        op=model_name+','+str(seq_length)+','+str(max_frames)+','+saved_model+','+acc
        f.write(op)
    return
    
if __name__=='__main__':
    main()