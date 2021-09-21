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
from dataLib.utils import LOG_INFO, correctPadding
import math
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
#--------------------------------
# vocab
#--------------------------------
with open("../vocab.json") as f:
    vocab = json.load(f)["vocab"]

start_end_value=len(vocab)
pad_value =start_end_value+1
LOG_INFO(f"start-end:{start_end_value}")
LOG_INFO(f"pad:{pad_value}")

#--------------------------------
# main
#--------------------------------


def pad_label(x,max_len):
    '''
        lambda function to create padded label for robust scanner
    '''
    x=[start_end_value]+x+[start_end_value]
    pad=[pad_value for _ in range(max_len-len(x))]
    return x+pad
    

def main(args):
    #-----------------
    # args
    #-----------------
    recog_dir=args.recog_dir
    img_height=int(args.img_height)
    img_width=int(args.img_width)
    factor=int(args.factor)
    max_len=int(args.max_len)
    #--- resource----------------
    img_dir =os.path.join(recog_dir,"images")
    data_csv =os.path.join(recog_dir,"data.csv")
    df=pd.read_csv(data_csv)
    #--- df processing-----------
    df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(img_dir,x))
    # unicodes
    df["unicodes"]=df.text.progress_apply(lambda x: [u for u in x])
    # encoding
    df["encoded"]=df.unicodes.progress_apply(lambda x:[vocab.index(u) for u in x])
    # length correction
    df["label_length"]=df.encoded.progress_apply(lambda x:len(x))

    LOG_INFO(f"Max Label Lenght:{max(df['label_length'].tolist())}")
    
    df["label_length"]=df["label_length"].progress_apply(lambda x: x if x < max_len else None)
    df.dropna(inplace=True)
    LOG_INFO(f"Max Label Lenght after correction:{max(df['label_length'].tolist())}")
    
    # label formation
    df["label"]=df.encoded.progress_apply(lambda x:pad_label(x,max_len))

    masks=[]
    #--- resize and pad images and create masks
    for idx in tqdm(range(len(df))):
        try:
            # image saving
            img_path=df.iloc[idx,2]
            img=cv2.imread(img_path)
            img,mask=correctPadding(img,(img_height,img_width),ptype="left",pvalue=255)
            cv2.imwrite(img_path,img)
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
    df=df[["img_path","label","mask","text"]]
    df.to_csv(data_csv,index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Recog Data Creation Script")
    parser.add_argument("recog_dir", help="Path to recog data")
    parser.add_argument("--img_height",required=False,default=64,help="height dimension to save images")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension to save images")
    parser.add_argument("--factor",required=False,default=32,help ="mask factor")
    parser.add_argument("--max_len",required=False,default=100,help ="max length to pad")
    args = parser.parse_args()
    main(args)