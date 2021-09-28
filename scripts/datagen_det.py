# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys

from cv2 import data
sys.path.append('../')

import argparse
from dataLib.utils import *
from dataLib.craft import gaussian_heatmap,det_data, pad_map
from tqdm.auto import tqdm
from ast import literal_eval
import os
import cv2
import random
import pandas as pd
tqdm.pandas()

def main(args):
    #-----------------
    card_dir    =   args.card_dir
    save_dir    =   args.save_dir
    save_dir    =   create_dir(save_dir,"detect")
    char_dir    =   create_dir(save_dir,"charmap")
    link_dir    =   create_dir(save_dir,"linkmap")
    img_dir     =   create_dir(save_dir,"image")

    data_csv    =   os.path.join(card_dir,"data.csv")
    mask_dir    =   os.path.join(card_dir,"masks")
    num_data    =   int(args.num_data)
    data_dim    =   int(args.data_dim)
    

    heatmap=gaussian_heatmap()
    # data division
    df=pd.read_csv(data_csv)
    cols=['bn_name', 'en_name', 'f_name', 'm_name', 'dob', 'nid']
    df["img_path"]=df["file"].progress_apply(lambda x:os.path.join(mask_dir,x))
    for col in cols:
        df[col]=df[col].progress_apply(lambda x:literal_eval(x))
    
    df["data_type"]=df["file"].progress_apply(lambda x: x.split("_")[0])
    df=df.sample(frac=1).reset_index(drop=True)
    nid_df  =df.loc[df["data_type"]=="nid"]
    smart_df=df.loc[df["data_type"]=="smart"]
    
    nid_df  =nid_df[:num_data]
    smart_df=smart_df[:num_data]
    nid_df=nid_df.sample(frac=1).reset_index(drop=True)
    smart_df=smart_df.sample(frac=1).reset_index(drop=True)
    
    for data_df in [nid_df,smart_df]:    
        for didx in tqdm(range(len(data_df))):
            #try:
            fname=data_df.iloc[didx,6]
            mask_path=data_df.iloc[didx,7]
            data_image_path=mask_path.replace("masks","images")
            # mask
            mask=cv2.imread(mask_path,0)
            # image
            image=cv2.imread(data_image_path)
            
            text_dict=data_df.iloc[[didx]].to_dict()
            labels=[]
            for col in cols:
                labels+=text_dict[col][didx]
            heat,link=det_data(mask,labels,heatmap)
            # save
            image,_=padDetectionImage(image)
            heat   =pad_map(heat)
            link   =pad_map(link)
            cv2.imwrite(os.path.join(img_dir,fname),cv2.resize(image,(data_dim,data_dim)))                 
            cv2.imwrite(os.path.join(char_dir,fname),cv2.resize(heat,(data_dim,data_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))
            cv2.imwrite(os.path.join(link_dir,fname),cv2.resize(link,(data_dim,data_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))    
            # except Exception as e:
            #     pass
    
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Segmentation Data Creation Script")
    parser.add_argument("card_dir", help="Path to cards data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--data_dim",required=False,default=1024,help="dimension of data to save the images")
    parser.add_argument("--num_data",required=False,default=10000,help ="number of data to create : default=10000")
    
    args = parser.parse_args()
    main(args)