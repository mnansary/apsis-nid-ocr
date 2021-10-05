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
from .segment import Extractor
from deepface import DeepFace
from .data import card

#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,
                model_dir,
                use_extractor=True,
                use_detector=True,
                use_recognizer=True,
                use_facematcher=True):
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
        '''
        
        # nid dummy image
        dummy_path=os.path.join(os.getcwd(),"tests","det.png")
        dummy_img=cv2.imread(dummy_path)
        dummy_boxes=[]
        for _,v in card.nid.front.box_dict.items():
            dummy_boxes.append(v)
        
        if use_extractor:
            try:
                ext_weights=os.path.join(model_dir,"segment.h5")
                self.extractor=Extractor(ext_weights)
                LOG_INFO("Extractor Loaded")    
                card_type,card_image=self.extractor.process(dummy_path)
                if card_type=="nid":
                    LOG_INFO("Extractor Initialized")
            except Exception as e:
                LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
            

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
                self.rec=RobustScanner(model_dir)
                LOG_INFO("Recognizer Loaded")
                texts=self.rec.recognize(dummy_img,dummy_boxes,
                                        batch_size=32,
                                        infer_len=10,
                                        word_process_func=None)
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

    def detect_boxes(self,img,det_thresh=0.4,text_thresh=0.7,shift_x_max=0):
        '''
            detection wrapper
            args:
                img         : the np.array format image to run detction on
                det_thresh  : detection threshold to use
                text_thresh : threshold for text data
                shift_x_max : pixels to shift x max  
            returns:
                boxes   :   returns boxes that contains text region
        '''
        boxes=self.det.detect(img,det_thresh=det_thresh,text_thresh=text_thresh,shift_x_max=shift_x_max)
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
        data=pd.DataFrame({"box":text_boxes})
        # detect field
        data["field"]=data.box.apply(lambda x:localize_box(x,region_boxes))
        data.dropna(inplace=True) 
        data["field"]=data["field"].apply(lambda x:region_fields[int(x)])
        
        box_dict={}
        df_box=[]
        df_field=[]
        for field in data.field.unique():
            df=data.loc[data.field==field]
            boxes=df.box.tolist()
            boxes=sorted(boxes, key=lambda x: x[0])
            box_dict[field]=boxes

            for box in boxes:
                df_box.append(box)
                df_field.append(field)
        
        df=pd.DataFrame({"box":df_box,"field":df_field})
        return box_dict,df 

    

    def extract(self,img,batch_size=32,shift_x_max=5,word_process_func=None):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                card_type           :   nid/smart card
                batch_size          :   batch size for inference
                word_process_func   :   function to process the word images
                shift_x_max         :   shifting x values
        '''
        # process if path is provided
        if type(img)==str:
            img=cv2.imread(img)
        # dims
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        card_type=self.extractor.process(img)
        
        if card_type=="nid": 
            src=card.nid.front
            two_step_recog=True
        else: 
            src=card.smart.front
            two_step_recog=False
        
        # face and sign
        img=cv2.resize(img,(card.width,card.height))
        x1,y1,x2,y2=src.face
        face=img[y1:y2,x1:x2]
        x1,y1,x2,y2=src.sign
        sign=img[y1:y2,x1:x2]
        img= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        # boxes
        text_boxes=self.detect_boxes(img,shift_x_max=shift_x_max)
        box_dict,df=self.process_boxes(text_boxes,src.box_dict)

        # recognition
        if two_step_recog:
            boxes=[]
            for k,v in box_dict.items():
                if k!="ID No.":
                    boxes+=v
            texts=self.rec.recognize(img,boxes,batch_size=batch_size,infer_len=10,word_process_func=word_process_func)
            #nid
            boxes=box_dict["ID No."]
            texts+=self.rec.recognize(img,boxes,batch_size=batch_size,infer_len=20,word_process_func=word_process_func)
        else:
            boxes=df.box.tolist()
            texts=self.rec.recognize(img,boxes,batch_size=batch_size,infer_len=10,word_process_func=word_process_func)
        
        response={}
        response["card_type"]=card_type
        df["text"]=texts
        for field in df.field.unique():
            tdf=df.loc[df.field==field]
            response[field]=" ".join(tdf.text.tolist())
        print(response.keys())
        # correct numbers
        if "o" in response["ID No."]:
            response["ID No."]=response["ID No."].replace("o","0")
        if "s" in response["ID No."]:
            response["ID No."]=response["ID No."].replace("s","5")
        if "z" in response["ID No."]:
            response["ID No."]=response["ID No."].replace("z","2")
        if "i" in response["ID No."]:
            response["ID No."]=response["ID No."].replace("i","1")
        if "l" in response["ID No."]:
            response["ID No."]=response["ID No."].replace("l","1")

        response["image"]=img2base64(face)
        response["sign"]=img2base64(sign)
        
        return response                  


