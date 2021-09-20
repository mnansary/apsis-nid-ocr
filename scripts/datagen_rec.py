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
from dataLib.utils import create_dir,LOG_INFO,remove_shadows,threshold_image
from tqdm.auto import tqdm
import os
import cv2
import numpy as np
import pandas as pd
tqdm.pandas()

def main(args):
    #-----------------
    card_dir=args.card_dir
    src_dir =args.src_dir
    save_dir=args.save_dir
    save_dir=create_dir(save_dir,"recog")
    img_dir =create_dir(save_dir,"images")
    data_csv =os.path.join(save_dir,"data.csv")
    
    src=Data(src_dir)
    LOG_INFO(save_dir)
    # data division
    card_img_dir =os.path.join(card_dir,"images")
    card_data_csv=os.path.join(card_dir,"data.csv")
    df=pd.read_csv(card_data_csv)
    df["img_path"]=df["file"].progress_apply(lambda x:os.path.join(card_img_dir,x))
    dicts=[]
    for didx in tqdm(range(len(df))):
        try:
            # construct row dictionary  
            text_dict=df.iloc[[didx]].to_dict()
            # img_path
            img_path=text_dict["img_path"][didx]
            iden    =text_dict["file"][didx].split(".")[0]
            # card type
            if "nid" in img_path:
                card_text=src.card.nid.front.text
            else:
                card_text=src.card.smart.front.text

            img=cv2.imread(img_path)
            # keep a scene copy
            org=np.copy(img)
            # text binary
            img=remove_shadows(img)
            img=threshold_image(img)
            h,w=img.shape
            # crop text and image data
            img_count=0
            for k,v in card_text.items():
                text=text_dict[k][didx]
                x1,y1,x2,y2=v["location"]
                crop=np.ones((h,w))
                crop[y1:y2,x1:x2]=img[y1:y2,x1:x2]
                
                crop=255-crop
                idx=np.where(crop==255)
                y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                crop=org[y_min:y_max,x_min:x_max]
                filename=f"{iden}_{img_count}.png"
                cv2.imwrite(os.path.join(img_dir,filename),crop)
                img_count+=1
                dicts.append({"filename":filename,"text":text})
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
    
    args = parser.parse_args()
    main(args)