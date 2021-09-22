#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import cv2
from .utils import LOG_INFO
from .recognizer import RobustScanner
from .data import card
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------
# model
#------------------------


class NIDOCR(object):
    def __init__(self,model_dir):
        # segment (To be added)

        # recog
        self.recognizer=RobustScanner(model_dir)
        LOG_INFO("Recognizer initialized")

    def detect(self,img):
        pass

    def extract(self,img,card_type):
        '''
            predict based on datatype
        '''
        if card_type=="nid": src=card.nid.front
        else: src=card.smart.front

        img=cv2.resize(img,(card.width,card.height))
        # face and sign
        x1,y1,x2,y2=src.face
        face=img[y1:y2,x1:x2]
        x1,y1,x2,y2=src.sign
        sign=img[y1:y2,x1:x2]
        # info
        boxes=[]
        infos=[]
        for k,v in src.box_dict.items():
            boxes.append(v)
            infos.append(k)
        texts=self.recognizer.recognize(img,boxes)
        data={
                "field":infos,
                "value":texts}
        info=pd.DataFrame(data)
        return face,sign,info                  