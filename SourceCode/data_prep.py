# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:21:48 2017

@Author: Vishveswaran Jothi
@Modifier:
Modified on Fri Nov 10 19:21:48 2017
"""

import util
import glob
from subprocess import call

def download_UCF101():
    """Download and extract the files if they dont exist. """
    fpath=os.path.dirname(os.path.dirname(__file__))+"/data/"
    file_name1="UCF101TrainTestSplits-RecognitionTask.zip"
    file_name2="UCF101.rar"
    fchk=fpath+"UCF101.rar"
    os.chdir(fpath)
    if not(os.path.exists(fchk)):        
        call('wget "http://crcv.ucf.edu/data/UCF101/UCF101.rar"',shell=True)        
        print("Download Complete...")
        call('wget "http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"',shell=True)
        print("Download Complete...")
    else:
        print("Data file exists hence not downloaded...")
    print("Extracting Data file if necessary...")
    extract(file_name1)
    print("Extract Complete...")
    extract(file_name2)
    print("Extract Complete...")    
    
    return
def extract(fname):
    extract_dir=os.path.abspath(fname.split('.')[0])
    f_type=fname.split('.')[1]
    if not(os.path.exists(extract_dir)):
        f_type="un"+f_type
        if f_type=="unrar":
            call([f_type,'e',fname])
        else:
            call([f_type,'-nq',fname])
        print("Files Extracted")
    else:
        print("File exist.. Hence not extracted")
    return

def dataset_prep(ver='01'):
    """This function moves all the files into the respective class folders with 
    the parent folder as either train or test according to the split that the 
    dataset belongs. It returns the file split as dictionary to use the result 
    as an input for converting videos into images"""
    fpath=os.path.dirname(os.path.dirname(__file__))+"/data/"
    os.chdir(fpath)
    train_file='./ucfTrainTestlist/trainlist' + ver + '.txt'
    test_file='./ucfTrainTestlist/testlist' + ver + '.txt'
    with open(test_file) as f:
        test_list = [row.strip() for row in list(f)]
    with open(train_file) as f:
        train_list = [(row.strip()).split(' ')[0] for row in list(f)]
    # Spliting dataset into training and testing
    file_split={'traindataset':train_list,'testdataset':test_list}
    for spt_name,file_list in file_split.items():
        for video in file_list:
            cls_name=video.split('/')[0]
            fname=video.split('/')[1]
            if not os.path.exists('./'+ spt_name+'/' + cls_name):    
                os.makedirs('./'+ spt_name+'/' + cls_name)
            if os.path.exists(fname):
                dest_path='./'+ spt_name+'/'+video
                call(['mv',fname,dest_path])
        else:
            print("Skipping %s file, since not found"%(fname))
    return 
def decode_video(datasets):
    """ This function is to decode the video into equence of frames with
    extension as jpg in the same folder as the video. The input for this 
    function is dictionary type which includes the split type and the files 
    associated with each split """
    #dat_content=[]
    # Check the file path
    op=""
    for grp in datasets:
        path='./'+grp+'/'
        if not os.path.exists('./'+grp+'/'):
            fpath=os.path.dirname(os.path.dirname(__file__))+"/data/"
            os.chdir(fpath)
        cls_folders=os.listdir(path)        
        for cls_f in cls_folders:
            cls_files=glob.glob(path+cls_f+'/*.avi')
            for video in cls_files:
                # Creating a database file to bookkeep the processed files for training and testing
                video_name=video.split('/')[3]
                frame_basename=video_name.split('.')[0]
                #check if the file is extracted or not
                if not os.path.exists(video.split('.avi')[0]+'_0001.jpg'):
                    dest=video.split('.avi')[0]+'_%04d.jpg'
                    call(["ffmpeg","-i",video,dest])
                # Getting frames per video
                no_frames=len(glob.glob(video.split('.avi')[0]+'*.jpg'))
                #dat_content.append([grp,cls_f,frame_basename,no_frames])
                op=op+grp+','+cls_f+','+frame_basename+','+str(no_frames)+'\n'
    #with open('Datainfo.csv','w') as f:
    #    fw=csv.writer(f)
    #    fw.writerows(dat_content)    
    with open('Datainfo.csv','w') as f:    
        f.write(op)
    return
if __name__=="__main__":
    download_UCF101()
    dataset_prep()
    decode_video(['traindataset','testdataset'])