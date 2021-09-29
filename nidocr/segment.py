#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import tensorflow as tf
import os
import numpy as np
import cv2 
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
from .utils import *
import matplotlib.pyplot as plt
#-------------------------
# model
#------------------------
class Extractor(object):
    def __init__(self,
                model_weights,
                img_dim=(256,256,3),
                data_channel=1,
                num_classes=2,
                labels=["nid","smart"],
                backbone='densenet121'):
        # Extractor with unet
        self.img_dim=img_dim
        self.data_channel=data_channel
        self.backbone=backbone
        self.num_classes=num_classes
        self.labels=labels
        
        unet=sm.Unet(self.backbone,input_shape=self.img_dim,encoder_weights=None,classes=self.data_channel)
        inp  =unet.input
        # class
        label=unet.get_layer(name="relu").output
        label = tf.keras.layers.GlobalAveragePooling2D()(label)
        label=tf.keras.layers.Dense(self.num_classes,activation="softmax",name="label")(label)
        # mask
        mask=unet.output
        self.model=tf.keras.Model(inputs=inp,outputs=[label,mask])
        # load weights        
        self.model.load_weights(model_weights)
    
    def process(self,img):
        # process if path is provided
        if type(img)==str:
            img=cv2.imread(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # dims
        h,w,d=img.shape
        # process
        img=cv2.resize(img,(self.img_dim[1],self.img_dim[0]))
        img=img/255.0
        data=np.expand_dims(img,axis=0)
        # predict
        pred=self.model.predict(data)
        card_type=self.labels[np.argmax(pred[0][0])]
        card_map =pred[1][0]
        card_map=cv2.resize(card_map,(w,h))
        # image
        card_image=convert_object(card_map,img)
        return card_type,card_image

