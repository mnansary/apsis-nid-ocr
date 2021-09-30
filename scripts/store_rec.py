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
DATA_NUM=10240

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def to_tfrecord(df,save_dir,r_num,iden):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        df
        df              :   portion of the whole dataframe to be saved	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
        iden            :   tfrec identifier 
    '''
    # record name
    tfrecord_name=f'{iden}_{r_num}.tfrecord'
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for idx in range(len(df)):
            image_path=df.iloc[idx,0]
            ulabel    =df.iloc[idx,1]
            glabel    =df.iloc[idx,2]
            mask      =df.iloc[idx,3]
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            
            data ={ 'image':_bytes_feature(image_bytes),
                    'ulabel':_int64_feature(ulabel),
                    'glabel':_int64_feature(glabel),
                    'mask':_int64_feature(mask)
            }
            
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(df,save_dir,iden):
    '''	        
        tf record wrapper
        args:	        
            df        :   dataframe that contains file and coord data	        
            save_dir  :   location to save the tfrecords	 
            iden      :   tfrec identifier   
    '''
    for i in tqdm(range(0,len(df),DATA_NUM)):
        # paths
        _df= df[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(_df,save_dir,r_num,iden)    


def main(args):
    #-----------------
    proc_dir=args.proc_dir
    save_dir=create_dir(proc_dir,"tfrecords")
    data_csv=os.path.join(proc_dir,"data.csv")
    df=pd.read_csv(data_csv)
    df.label_unicode=df.label_unicode.progress_apply(lambda x: literal_eval(x))
    df.label_grapheme=df.label_grapheme.progress_apply(lambda x: literal_eval(x))
    df["mask"]=df["mask"].progress_apply(lambda x: literal_eval(x))
    
    for source in df.source.unique():
        sdf=df.loc[df.source==source]
        genTFRecords(sdf,save_dir,source)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Recognition TFRecords Data Creation Script")
    parser.add_argument("proc_dir", help="Path to processed folder data")
    args = parser.parse_args()
    main(args)