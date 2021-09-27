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
from dataLib.utils import create_dir
from dataLib.craft import gaussian_heatmap,det_data
from tqdm.auto import tqdm
from ast import literal_eval
import os
import cv2
import random
import pandas as pd
tqdm.pandas()

def main(args):
    #-----------------
    card_dir=args.card_dir
    char_dir =create_dir(card_dir,"charmap")
    link_dir =create_dir(card_dir,"linkmap")
    data_csv =os.path.join(card_dir,"data.csv")
    mask_dir =os.path.join(card_dir,"masks")

    heatmap=gaussian_heatmap()
    # data division
    df=pd.read_csv(data_csv)
    cols=['bn_name', 'en_name', 'f_name', 'm_name', 'dob', 'nid']
    df["img_path"]=df["file"].progress_apply(lambda x:os.path.join(mask_dir,x))
    for col in cols:
        df[col]=df[col].progress_apply(lambda x:literal_eval(x))
    
    

    for didx in tqdm(range(len(df))):
        try:
            fname=df.iloc[didx,6]
            mask_path=df.iloc[didx,7]
            mask=cv2.imread(mask_path,0)
            text_dict=df.iloc[[didx]].to_dict()
            labels=[]
            for col in cols:
                labels+=text_dict[col][didx]
            heat,link=det_data(mask,labels,heatmap)
            # save
            cv2.imwrite(os.path.join(char_dir,fname),heat)
            cv2.imwrite(os.path.join(link_dir,fname),link)    
        except Exception as e:
            pass
    
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Detection Data Creation Script")
    parser.add_argument("card_dir", help="Path to cards data")
    
    args = parser.parse_args()
    main(args)