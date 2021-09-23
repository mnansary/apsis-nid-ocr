#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt 
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr

def locateData(img,val):
    '''
        locates img data based on specific value threshold
    '''
    idx=np.where(img>val)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    return y_min,y_max,x_min,x_max

def remove_shadows(img):
    '''
    removes shadows and thresholds
    '''
    assert len(img.shape)==3
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result_norm_planes = []
    # split rgb
    rgb_planes = cv2.split(img)
    # clean planes
    for plane in rgb_planes:
        # dilate
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # background
        bg_img = cv2.medianBlur(dilated_img, 21)
        # difference
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # normalized
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # append
        result_norm_planes.append(norm_img)
    # merge rgb
    img = cv2.merge(result_norm_planes)
    return img

def threshold_image(img,blur):
    '''
        threshold an image
    '''
    assert len(img.shape)==3
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    if blur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def cleanImage(img,remove_shadow=True):
    '''
        cleans an image 
    '''
    # text binary
    if remove_shadow:
        img=remove_shadows(img)
    img=threshold_image(img,blur=True)
    # remove noise
    img=cv2.merge((img,img,img))
    img= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return img

def enhanceImage(img,factor=10):
    '''
        enhances an image based on contrast
    '''
    img=Image.fromarray(img)
    con_enhancer = ImageEnhance.Contrast(img)
    img= con_enhancer.enhance(factor)
    img=np.array(img)
    return img 


#---------------------------------------------------------------
# detection utils
#---------------------------------------------------------------
def padDetectionImage(img):
    cfg={}
        
    h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        pad =np.ones((h,pad_width,d))*255
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        pad =np.ones((pad_height,w,d))*255
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    return img.astype("uint8"),cfg
   
#---------------------------------------------------------------
# modifier/Recognition util utils
#---------------------------------------------------------------
def padData(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8") 

def padWords(img,dim,ptype="central",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padData(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padData(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
    

#------------------------------------
# region-utils 
#-------------------------------------
def intersection(boxA, boxB):
    # boxA=ref
    # boxB=sig
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    x_min,y_min,x_max,y_max=boxB
    selfArea  = abs((y_max-y_min)*(x_max-x_min))
    return interArea/selfArea

def localize_box(box,region_boxes):
    '''
        lambda localization
    '''
    max_ival=0
    box_id=None
    for idx,region_box in enumerate(region_boxes):
        ival=intersection(region_box,box)
        if ival==1:
            return idx
        if ival>max_ival:
            max_ival=ival
            box_id=idx
    if max_ival==0:
        return None
    return box_id

#----------------------------------------
# display utils
#----------------------------------------
def display_data(info,img,cv_color=True):
    if type(img)=="str":
        img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        if cv_color:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    LOG_INFO("------------------------------------------------")
    LOG_INFO(f"{info}")
    LOG_INFO("------------------------------------------------")
    plt.imshow(img)
    plt.show()
