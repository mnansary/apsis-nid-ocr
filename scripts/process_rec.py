# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
from genericpath import sameopenfile
import sys
sys.path.append('../')

import argparse
import cv2
import os
import json
import numpy as np
from dataLib.utils import LOG_INFO, correctPadding,GraphemeParser, create_dir
import math
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
#--------------------------------
# main
#--------------------------------
GP=GraphemeParser()


def pad_label(x,max_len,pad_value,start_end_value):
    '''
        lambda function to create padded label for robust scanner
    '''
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
    recog_dir=args.recog_dir
    save_dir=create_dir(recog_dir,"processed")
    img_save_dir=create_dir(save_dir,"images")
    img_height=int(args.img_height)
    img_width=int(args.img_width)
    factor=int(args.factor)
    max_len=int(args.max_len)
    #--- resource----------------
    img_dir =os.path.join(recog_dir,"images")
    data_csv =os.path.join(recog_dir,"data.csv")
    df=pd.read_csv(data_csv)
    df.dropna(inplace=True)
    
    #--------------------------------
    # vocab
    #--------------------------------
    with open("../vocab.json") as f:
        vocab = json.load(f)
    uvocab =vocab["unicode"]
    gvocab=vocab["grapheme"]

    
    #--- df processing-----------
    df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(img_dir,x))
    # unicodes
    df["unicodes"]=df.text.progress_apply(lambda x: [u for u in x])
    # graphemes
    df["graphemes"]=df.text.progress_apply(lambda x: GP.word2grapheme(x))
    # encoding
    df["encoded_unicodes"]=df.unicodes.progress_apply(lambda x:encode_label(x,uvocab))
    df.dropna(inplace=True)
    df["encoded_graphemes"]=df.graphemes.progress_apply(lambda x:encode_label(x,gvocab))
    df.dropna(inplace=True)
    # label formation
    LOG_INFO("Unicode")
    start_end_value=len(uvocab)
    pad_value =start_end_value+1
    LOG_INFO(f"start-end:{start_end_value}")
    LOG_INFO(f"pad:{pad_value}")
    df["label_unicode"]=df.encoded_unicodes.progress_apply(lambda x:pad_label(x,max_len,pad_value,start_end_value))

    LOG_INFO("Graphemes")
    start_end_value=len(gvocab)+1
    pad_value =start_end_value+1
    LOG_INFO(f"start-end:{start_end_value}")
    LOG_INFO(f"pad:{pad_value}")
    df["label_grapheme"]=df.encoded_graphemes.progress_apply(lambda x:pad_label(x,max_len,pad_value,start_end_value))

    masks=[]
    #--- resize and pad images and create masks
    for idx in tqdm(range(len(df))):
        try:
            # image saving
            img_path=df.iloc[idx,2]
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
    df=df[["img_path","label_unicode","label_grapheme","mask","text"]]
    df.to_csv(os.path.join(save_dir,"data.csv"),index=False)

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