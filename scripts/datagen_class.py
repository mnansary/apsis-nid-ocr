# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
from dataLib.data import Data
from dataLib.segment import render_data
from dataLib.utils import create_dir,LOG_INFO
from tqdm.auto import tqdm
import os
import cv2
import random
import pandas as pd
tqdm.pandas()

def main(args):
    #-----------------
    card_dir=args.card_dir
    src_dir =args.src_dir
    save_dir=args.save_dir
    save_dir=create_dir(save_dir,"classification")
    img_dir =create_dir(save_dir,"images")
    data_csv =os.path.join(save_dir,"data.csv")
    num_data=int(args.num_data)
    data_dim=int(args.data_dim)
    
    src=Data(src_dir)
    LOG_INFO(save_dir)
    backgen=src.backgroundGenerator()
    # data division
    card_img_dir =os.path.join(card_dir,"images")
    card_data_csv=os.path.join(card_dir,"data.csv")
    df=pd.read_csv(card_data_csv)
    df=df[["file"]]
    df["data_type"]=df["file"].progress_apply(lambda x: x.split("_")[0])
    df["img_path"]=df["file"].progress_apply(lambda x: os.path.join(card_img_dir,x))
    df=df[["img_path","data_type"]]
    df=df.sample(frac=1)
    nid_df  =df.loc[df["data_type"]=="nid"]
    smart_df=df.loc[df["data_type"]=="smart"]
    
    nid_df  =nid_df[:num_data]
    smart_df=smart_df[:num_data]
    nid_df=nid_df.sample(frac=1).reset_index(drop=True)
    smart_df=smart_df.sample(frac=1).reset_index(drop=True)
    

    dicts=[]

    for data_df in [nid_df,smart_df]:
        for idx in tqdm(range(len(data_df))):
            try:
                data={}
                img_path =data_df.iloc[idx,0]
                card_type=data_df.iloc[idx,1]
                img,_,_=render_data(backgen,img_path,src.config)
                img=cv2.resize(img,(data_dim,data_dim))
                # save
                cv2.imwrite(os.path.join(img_dir,f"{card_type}_{idx}.png"),img)
                data["file"]=f"{card_type}_{idx}.png"
                dicts.append(data)    
            except Exception as e:
                pass
    df=pd.DataFrame(dicts)
    df.to_csv(data_csv,index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Classification Data Creation Script")
    parser.add_argument("src_dir", help="Path to source data")
    parser.add_argument("card_dir", help="Path to cards data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--data_dim",required=False,default=256,help="dimension of data to save the images")
    parser.add_argument("--num_data",required=False,default=10000,help ="number of data to create : default=10000")
    
    args = parser.parse_args()
    main(args)