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
from dataLib.utils import create_dir
from tqdm.auto import tqdm
from glob import glob
import os
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

def to_tfrecord(img_paths,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        
        img_path        :   portion of the whole paths to be saved	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in img_paths:
            char_path =str(image_path).replace('image','charmap')
            link_path =str(image_path).replace('image','linkmap')
            
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            # charmap
            with(open(char_path,'rb')) as fid:
                char_bytes=fid.read()
            
            # linkmap
            with(open(link_path,'rb')) as fid:
                link_bytes=fid.read()
            
            data ={ 'image':_bytes_feature(image_bytes),
                    'charmap':_bytes_feature(char_bytes),
                    'linkmap':_bytes_feature(link_bytes)
            }
            
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(img_paths,save_dir):
    '''	        
        tf record wrapper
        args:	        
            img_paths   :   img_paths to store        
            save_dir    :   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(img_paths),DATA_NUM)):
        # paths
        _paths= img_paths[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(_paths,save_dir,r_num)    



def main(args):
    #-----------------
    det_dir=args.det_dir
    save_dir=create_dir(det_dir,"tfrecords")
    
    img_dir =os.path.join(det_dir,"image")
    img_paths=[img_path for img_path in tqdm(glob(os.path.join(img_dir,"*.*")))]
    genTFRecords(img_paths,save_dir)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Detection TFRecords Data Creation Script")
    parser.add_argument("det_dir", help="Path to detection folder data")
    args = parser.parse_args()
    main(args)