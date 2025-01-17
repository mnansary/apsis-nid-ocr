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
import cv2
import os
import json
import numpy as np
from dataLib.utils import *
import math
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
#--------------------------------
# main
#--------------------------------
GP=GraphemeParser(language.bangla)


def pad_label(x,max_len,pad_value,start_end_value):
    '''
        lambda function to create padded label for robust scanner
    '''
    if len(x)>max_len-2:
        return None
    else:
        x=[start_end_value]+x+[start_end_value]
        pad=[pad_value for _ in range(max_len-len(x))]
        return x+pad
        
def encode_label(x,vocab):
    label=[]
    for ch in x:
        try:
            label.append(vocab.index(ch))
        except Exception as e:
            return None
    return label

def main(args):
    #-----------------
    # args
    #-----------------
    recog_dir   =   args.recog_dir
    save_dir    =   args.save_dir
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    factor      =   int(args.factor)
    max_len     =   int(args.max_len)
    use_lower   =   args.use_lower_only
    #--- resource----------------
    save_dir    =   create_dir(save_dir,"processed")
    img_save_dir=   create_dir(save_dir,"images")
    
    img_dir =os.path.join(recog_dir,"images")
    data_csv =os.path.join(recog_dir,"data.csv")
    
    # dataframe
    df=pd.read_csv(data_csv)
    df.dropna(inplace=True)
    
    #--------------------------------
    # vocab
    #--------------------------------
    with open("../vocab.json") as f:
        vocab = json.load(f)["vocab"]
    
    
    #--- df processing-----------
    df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(img_dir,x))

    if use_lower:
        df.text=df.text.progress_apply(lambda x:x.lower())
    # unicodes
    df["components"]=df.text.progress_apply(lambda x: GP.process(x))
    df["components"]=df.components.progress_apply(lambda x: None if len(x)>max_len-2 else x)
    df.dropna(inplace=True)
    # encoding
    df["encoded"]=df.components.progress_apply(lambda x:encode_label(x,vocab))
    df.dropna(inplace=True)
    # label formation
    LOG_INFO("Components: Start null, end null")
    start_end_value=len(vocab)+1
    pad_value =len(vocab)+2
    LOG_INFO(f"start-end:{start_end_value}")
    LOG_INFO(f"pad:{pad_value}")
    df["label"]=df.encoded.progress_apply(lambda x:pad_label(x,max_len,pad_value,start_end_value))
    df.dropna(inplace=True)
    df["len"]=df.label.progress_apply(lambda x:len(x))
    LOG_INFO(f"Max Len:{max(df['len'].tolist())}")
    
    masks=[]
    #--- resize and pad images and create masks
    for idx in tqdm(range(len(df))):
        try:
            # image saving
            img_path=df.iloc[idx,3]
            fname=os.path.basename(img_path)
            img=cv2.imread(img_path)
            img,mask=correctPadding(img,(img_height,img_width),ptype="left",pvalue=255)
            cv2.imwrite(os.path.join(img_save_dir,fname),img)
            # mask
            mask=math.ceil((mask/img_width)*(img_width//factor))
            imask=np.zeros((img_height//factor,img_width//factor))
            imask[:,:mask]=1
            imask=imask.flatten().tolist()
            imask=[int(i) for i in imask]
            masks.append(imask)
        except Exception as e:
            pass
            masks.append(None)
    
    df["mask"]=masks
    df.dropna(inplace=True)
    
    df=df[["img_path","label","mask","text","source"]]
    df["img_path"]=df["img_path"].progress_apply(lambda x: x.replace("recog","processed"))
    df.to_csv(os.path.join(save_dir,"data.csv"),index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Recog Data Creation Script")
    parser.add_argument("recog_dir", help="Path to recog data")
    parser.add_argument("save_dir", help="Path to save processed data")
    parser.add_argument("--img_height",required=False,default=64,help="height dimension to save images")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension to save images")
    parser.add_argument("--factor",required=False,default=32,help ="mask factor")
    parser.add_argument("--max_len",required=False,default=40,help ="max length to pad")
    parser.add_argument("--use_lower_only",required=False,type=str2bool,default=False,help ="use lowercase words")
    args = parser.parse_args()
    main(args)