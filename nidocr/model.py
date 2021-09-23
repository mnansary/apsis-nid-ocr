#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import cv2
from .utils import *
import pandas as pd
import matplotlib.pyplot as plt
from .detector import CRAFT
from .recogonizer import RobustScanner
from deepface import DeepFace
from .data import card
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,
                model_dir,
                use_detector=True,
                use_recognizer=True,
                use_facematcher=True,
                rec_img_process_func=None,
                rec_word_process_func=None,
                rec_infer_len=20,
                rec_batch_size=32):
        '''
            Instantiates an ocr model:
                methods:
                detect_boxes    :   runs detector and returns boxes where text exists
                process_boxes   :   processes regional bboxes

            args:
                model_dir               :   path of the model weights
                use_detector            :   flag for loading detector model
                use_recognizer          :   flag for loading recognizer model
                use_facematcher         :   flag for loading facematcher model
                rec_img_process_func    :   custom function for recognizer reference image processing
                rec_word_process_func   :   custom function for recognizer word image processing
                rec_infer_len           :   inference loop length
                rec_batch_size          :   batch size for inference
        '''
        
        # nid dummy image
        dummy_path=os.path.join(os.getcwd(),"tests","det.png")
        dummy_img=cv2.imread(dummy_path)
        dummy_boxes=[]
        for _,v in card.nid.front.box_dict.items():
            dummy_boxes.append(v)
        
        # detector weight loading and initialization
        if use_detector:
            try:
                craft_weights=os.path.join(model_dir,"craft.h5")
                self.det=CRAFT(craft_weights)
                
                LOG_INFO("Detector Loaded")    
                boxes=self.det.detect(dummy_img)
                if len(boxes)>0:
                    LOG_INFO("Detector Initialized")
            except Exception as e:
                LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
            

        # recognizer weight loading and initialization
        if use_recognizer:
            try:
                self.rec_img_process_func =rec_img_process_func
                self.rec_word_process_func=rec_word_process_func
                self.rec_infer_len        =rec_infer_len
                self.rec_batch_size       =rec_batch_size
                self.rec=RobustScanner(model_dir)
                LOG_INFO("Recognizer Loaded")
                texts=self.rec.recognize(dummy_img,dummy_boxes,
                                        batch_size=self.rec_batch_size,
                                        infer_len=self.rec_infer_len,
                                        img_process_func=self.rec_img_process_func,
                                        word_process_func=self.rec_word_process_func)
                if len(texts)>0:
                    LOG_INFO("Recognizer Initialized")

            except Exception as e:
                LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        # facematcher loading and initialization
        if use_facematcher:
            try:
                obj=DeepFace.verify(dummy_path,dummy_path,model_name = 'ArcFace', detector_backend = 'retinaface')
                if obj['verified']:
                    LOG_INFO("Face Matcher Initialized")
            except Exception as e:
                LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")



    def facematcher(self,src_img,dest_img,return_dict=False):
        '''
            matches two given faces:
            args:
                src_img     :   the given id image (np.array)
                dest_img    :   the selfie/taken image to match (np.array)
                return_dict :   if set to True returns a dictionary of {match,similiarity} 
        '''
        src=os.path.join(os.getcwd(),"tests","src.png")
        dst=os.path.join(os.getcwd(),"tests","dst.png")
        cv2.imwrite(src,src_img)
        cv2.imwrite(dst,dest_img)
        obj=DeepFace.verify(src,dst,model_name = 'ArcFace', detector_backend = 'retinaface')
        
        match=obj["verified"]
        similiarity_value=round(obj['distance']*100,2)
        
        if match:            
            LOG_INFO(f"FOUND MATCH:  Similiarity ={similiarity_value}%")
        else:
            LOG_INFO(f"MATCH NOT FOUND:  Similiarity ={similiarity_value}%")
        if return_dict:
            return {"match":match,"similiarity":similiarity_value}

    def detect_boxes(self,img,card_height=614,card_width=1024,det_thresh=0.4,text_thresh=0.7):
        '''
            detection wrapper
            args:
                img         : the np.array format image to run detction on
                card_height : base height of a card (used for resizing)
                card_width  : base width of a card  (used for resizing)
                det_thresh  : detection threshold to use
                text_thresh : threshold for text data
            returns:
                boxes   :   returns boxes that contains text region
        '''
        img=cv2.resize(img,(card_width,card_height))
        img=cleanImage(img)
        boxes=self.det.detect(img,det_thresh=det_thresh,text_thresh=text_thresh)
        return boxes
    
    def process_boxes(self,text_boxes,region_dict):
        '''
            keeps relevant boxes with respect to region
            args:
                text_boxes  :  detected text boxes by the detector
                region_dict :  key,value pair dictionary of region_bbox and field info 
                               => {"field_name":[x_min,y_min,x_max,y_max]}
        '''
        # extract region boxes
        region_boxes=[]
        region_fields=[]
        for k,v in region_dict.items():
            region_fields.append(k)
            region_boxes.append(v)
        # sort boxed
        text_boxes=sorted(text_boxes,key=lambda k: [k[1], k[0]])
        data=pd.DataFrame({"box":text_boxes})
        # detect field
        data["field"]=data.box.apply(lambda x:localize_box(x,region_boxes))
        data.dropna(inplace=True) 
        data["field"]=data["field"].apply(lambda x:region_fields[int(x)])
        return data 

    

    # def extract(self,img,card_type,batch_size=32):
    #     '''
    #         predict based on datatype
    #     '''
    #     if card_type=="nid": src=card.nid.front
    #     else: src=card.smart.front

    #     img=cv2.resize(img,(card.width,card.height))
    #     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     # face and sign
    #     x1,y1,x2,y2=src.face
    #     face=img[y1:y2,x1:x2]
    #     x1,y1,x2,y2=src.sign
    #     sign=img[y1:y2,x1:x2]
    #     # info
    #     boxes=[]
    #     infos=[]
    #     for k,v in src.box_dict.items():
    #         boxes.append(v)
    #         infos.append(k)
    #     texts=self.recognizer.recognize(img,boxes,batch_size=batch_size)
    #     data={
    #             "field":infos,
    #             "value":texts}
    #     info=pd.DataFrame(data)
    #     return face,sign,info                  