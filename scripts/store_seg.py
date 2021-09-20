# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')
import tensorflow as tf 
import argparse
from dataLib.utils import create_dir,LOG_INFO
from tqdm.auto import tqdm
import os
import pandas as pd
from ast import literal_eval
tqdm.pandas()
#--------------------
# utils
#--------------------
DATA_NUM=128

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def to_tfrecord(df,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        df
        df              :   portion of the whole dataframe to be saved	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for idx in range(len(df)):
            image_path=df.iloc[idx,0]
            coord     =df.iloc[idx,1]
            if "nid" in image_path:
                label     =[1,0]
            else:
                label     =[0,1]
            mask_path =str(image_path).replace('images','masks')
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            # maskdf
            with(open(mask_path,'rb')) as fid:
                mask_bytes=fid.read()
            
            data ={ 'image':_bytes_feature(image_bytes),
                    'mask':_bytes_feature(mask_bytes),
                    'label':_int64_feature(label),
                    'bbox':_int64_feature(coord)
            }
            
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(df,save_dir):
    '''	        
        tf record wrapper
        args:	        
            df        :   dataframe that contains file and coord data	        
            save_dir  :   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(df),DATA_NUM)):
        # paths
        _df= df[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(_df,save_dir,r_num)    

def flat_coord(x):
    '''
        returns flat coord data 
    '''
    flat=[]
    for c in x:
        flat+=c
    return flat


def main(args):
    #-----------------
    seg_dir=args.seg_dir
    save_dir=create_dir(seg_dir,"tfrecords")
    
    img_dir =os.path.join(seg_dir,"images")
    data_csv=os.path.join(seg_dir,"data.csv")
    df=pd.read_csv(data_csv)
    df.coord=df.coord.progress_apply(lambda x: literal_eval(x))
    df.file =df.file.progress_apply(lambda x: os.path.join(img_dir,x))
    df.coord=df.coord.progress_apply(lambda x: flat_coord(x))
    genTFRecords(df,save_dir)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Segmentation TFRecords Data Creation Script")
    parser.add_argument("seg_dir", help="Path to segment folder data")
    args = parser.parse_args()
    main(args)