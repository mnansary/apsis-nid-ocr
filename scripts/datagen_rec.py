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
    aug=Modifier(use_gaussnoise=True,use_bifilter=True)
    card_dir=args.card_dir
    src_dir =args.src_dir
    save_dir=args.save_dir
    save_dir=create_dir(save_dir,"recog")    
    img_dir =create_dir(save_dir,"images")
    data_csv =os.path.join(save_dir,"data.csv")
    gen_scene=args.pure_scene_text
    src=Data(src_dir)
    backgen=src.backgroundGenerator(dim=(256,256))
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
    
    eng_keys=["en_name","dob","nid"]
        
    img_count=0
    dicts=[]
    # remove later
    df=df.sample(frac=1)
    df=df[:10000]

    for didx in tqdm(range(len(df))):
        try:
            # construct row dictionary  
            text_dict=df.iloc[[didx]].to_dict()
            # img_path
            img_path=text_dict["img_path"][didx]
            mask_path=img_path.replace("images","masks")
            # card type
            if "nid" in img_path:
                card_text=src.card.nid.front.text
            else:
                card_text=src.card.smart.front.text
            if not gen_scene:
                # image    
                img=cv2.imread(img_path)
                img=remove_shadows(img)
            # mask
            mask=cv2.imread(mask_path,0)
            
            for k,v in card_text.items():
                if k not in eng_keys:
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
                        if gen_scene:
                            word_mask=mask[y_min:y_max,x_min:x_max]
                            word_img=next(backgen)
                            h,w=word_mask.shape
                            word_img=cv2.resize(word_img,(w,h))
                            word_img[word_mask>0]=randColor()
                            word_img=aug.noise(word_img)
                        else:
                            word_img=img[y_min:y_max,x_min:x_max]
                        
                        filename=f"{img_count}.png"
                        cv2.imwrite(os.path.join(img_dir,filename),word_img)
                        img_count+=1
                        dicts.append({"filename":filename,"text":text_word,"source":k})
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
    parser.add_argument("--pure_scene_text",required=False,default=False,type=str2bool,help ="generate pure scene text")
    args = parser.parse_args()
    main(args)