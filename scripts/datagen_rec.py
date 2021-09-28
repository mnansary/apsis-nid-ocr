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
from ast import literal_eval
from dataLib.data import Data
from dataLib.utils import *

from tqdm.auto import tqdm
import os
import cv2
import numpy as np
import pandas as pd
tqdm.pandas()

def main(args):
    #-----------------
    mod=Modifier()
    card_dir=args.card_dir
    src_dir =args.src_dir
    save_dir=args.save_dir
    use_noise=args.add_noise
    save_dir=create_dir(save_dir,"recog")
    if use_noise:
        save_dir=create_dir(save_dir,"noisy")
    else:
        save_dir=create_dir(save_dir,"clean")
        
    img_dir =create_dir(save_dir,"images")
    data_csv =os.path.join(save_dir,"data.csv")
    
    src=Data(src_dir)
    LOG_INFO(save_dir)
    # data division
    card_img_dir =os.path.join(card_dir,"images")
    card_mask_dir=os.path.join(card_dir,"masks")
    card_data_csv=os.path.join(card_dir,"data.csv")
    df=pd.read_csv(card_data_csv)
    cols=['bn_name', 'en_name', 'f_name', 'm_name', 'dob', 'nid']
    df["img_path"]=df["file"].progress_apply(lambda x:os.path.join(card_img_dir,x))
    for col in cols:
        df[col]=df[col].progress_apply(lambda x:literal_eval(x))

    img_count=0
    dicts=[]
    for didx in tqdm(range(len(df))):
        try:
            # construct row dictionary  
            text_dict=df.iloc[[didx]].to_dict()
            # img_path
            img_path=text_dict["img_path"][didx]
            mask_path=img_path.replace("images","masks")
            iden    =text_dict["file"][didx].split(".")[0]
            # card type
            if "nid" in img_path:
                card_text=src.card.nid.front.text
            else:
                card_text=src.card.smart.front.text
            # image    
            img=cv2.imread(img_path)
            if use_noise:
                img=mod.noise(img)
            # mask
            mask=cv2.imread(mask_path,0)
            # crop text and image data
           
            for k,v in card_text.items():
                text=text_dict[k][didx]
                for word in text:
                    text_word=''
                    mask_word=np.zeros_like(mask)
                    for kt,vt in word.items():
                        text_word+=vt
                        mask_word[mask==kt]=255
                    # word img crop
                    idx=np.where(mask_word==255)
                    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                    word_img=img[y_min:y_max,x_min:x_max]    
                    filename=f"{img_count}.png"
                    cv2.imwrite(os.path.join(img_dir,filename),word_img)
                    img_count+=1
                    dicts.append({"filename":filename,"text":text_word.lower()})
        except Exception as e:
            pass
    df=pd.DataFrame(dicts)
    df.to_csv(data_csv,index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Segmentation Data Creation Script")
    parser.add_argument("src_dir", help="Path to source data")
    parser.add_argument("card_dir", help="Path to cards data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--add_noise",type=str2bool,required=False,default=True,help ="using noise to create word data : default=True")    
    args = parser.parse_args()
    main(args)