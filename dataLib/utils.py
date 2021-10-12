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
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance
import argparse
#---------------------------------------------------------------
# common utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
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
def randColor():
    '''
        generates random color
    '''
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match
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
#---------------------------------------------------------------
def remove_shadows(image: np.ndarray):
    # split the image channels into b,g,r
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    rgb_planes = [b,g,r]

    # iniatialising the final shadow free normalised image list for planes
    result_norm_planes = []

    # removing the shadows in individual planes
    for plane in rgb_planes:
        # dialting the image to spead the text to the background
        dilated_image = cv2.dilate(plane, np.ones((7,7), np.uint8))
        
        # blurring the image to get the backround image
        bg_image = cv2.medianBlur(dilated_image, 21)

        # subtracting the plane-background from the image-plane
        diff_image = 255 - cv2.absdiff(plane, bg_image)

        # normalisng the plane
        norm_image = cv2.normalize(diff_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # appending the plane to the final planes list
        result_norm_planes.append(norm_image)

    # merging the shadow-free planes into one image
    normalised_image = cv2.merge(result_norm_planes)

    # returning the normalised image
    return normalised_image
#---------------------------------------------------------------
def threshold_image(img,blur=True):
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
#---------------------------------------------------------------
def cleanImage(img,remove_shadow=True,blur=True):
    '''
        cleans an image 
    '''
    # text binary
    if remove_shadow:
        img=remove_shadows(img)
    img=threshold_image(img,blur=blur)
    # remove noise
    img=cv2.merge((img,img,img))
    img= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return img
#---------------------------------------------------------------
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
def padToFixedHeightWidth(img,h_max,w_max):
    '''
        pads an image to fixed height and width
    '''
    # shape
    h,w=img.shape
    if w<w_max:
        # pad widths
        pad_width =(w_max-w)        
        pad =np.zeros((h,pad_width))
        # pad
        img =np.concatenate([img,pad],axis=1)
    elif w>w_max: # reduce height
        h_new=int(w_max*h/w)
        img = cv2.resize(img, (w_max,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # shape
    h,w=img.shape
    if h<h_max:    
        # pad heights
        pad_height_top =1+(h_max-h)//2
        pad_height_bot =1+h_max-h-pad_height_top
                
        pad_top =np.zeros((pad_height_top,w))
        pad_bot =np.zeros((pad_height_bot,w))
        # pad
        img =np.concatenate([pad_top,img,pad_bot],axis=0)
    elif h>h_max:
        w_new=int(h_max*w/h)
        img = cv2.resize(img, (w_new,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    img = cv2.resize(img, (w_max,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------recog utils-----------------------------------------------------
def padWordImage(img,pad_loc,pad_dim,pad_type,pad_val):
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
#---------------------------------------------------------------
def correctPadding(img,dim,ptype="central",pvalue=255):
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
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#----------------------------------------
# noise utils
#----------------------------------------
class Modifier:
    def __init__(self,
                blur_kernel_size_max=6,
                blur_kernel_size_min=3,
                bi_filter_dim_min=7,
                bi_filter_dim_max=12,
                bi_filter_sigma_max=80,
                bi_filter_sigma_min=70):

        self.blur_kernel_size_max   =   blur_kernel_size_max
        self.blur_kernel_size_min   =   blur_kernel_size_min
        self.bi_filter_dim_min      =   bi_filter_dim_min
        self.bi_filter_dim_max      =   bi_filter_dim_max
        self.bi_filter_sigma_min    =   bi_filter_sigma_min
        self.bi_filter_sigma_max    =   bi_filter_sigma_max
        
    def __initParams(self):
        self.blur_kernel_size=random.randrange(self.blur_kernel_size_min,
                                               self.blur_kernel_size_max, 
                                               2)
        self.bi_filter_dim   =random.randrange(self.bi_filter_dim_min,
                                               self.bi_filter_dim_max, 
                                               2)
        self.bi_filter_sigma =random.randint(self.bi_filter_sigma_min,
                                             self.bi_filter_sigma_max)
        self.ops             =   [  self.__blur,
                                    self.__gaussBlur,
                                    self.__medianBlur,
                                    self.__biFilter,
                                    self.__gaussNoise,
                                    self.__addBrightness]


    def __blur(self,img):
        return cv2.blur(img,
                        (self.blur_kernel_size,
                        self.blur_kernel_size),
                         0)
    def __gaussBlur(self,img):
        return cv2.GaussianBlur(img,
                                (self.blur_kernel_size,
                                self.blur_kernel_size),
                                0) 
    def __medianBlur(self,img):
        return  cv2.medianBlur(img,
                               self.blur_kernel_size)
    def __biFilter(self,img):
        return cv2.bilateralFilter(img,
                                   self.bi_filter_dim,
                                   self.bi_filter_sigma,
                                   self.bi_filter_sigma)

    def __gaussNoise(self,image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        image = image+gauss
        return image.astype("uint8")
    
    def __addBrightness(self,image):    
        ## Conversion to HLSmask
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)     
        image_HLS = np.array(image_HLS, dtype = np.float64)
        ## generates value between 0.5 and 1.5       
        random_brightness_coefficient = np.random.uniform()+0.5  
        ## scale pixel values up or down for channel 1(Lightness) 
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
        ##Sets all values above 255 to 255    
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255     
        image_HLS = np.array(image_HLS, dtype = np.uint8)    
        ## Conversion to RGB
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)     
        return image_RGB
    
    def noise(self,img):
        self.__initParams()
        img=img.astype("uint8")
        idx = random.choice(range(len(self.ops)))
        img = self.ops.pop(idx)(img)
        return img
#--------------------
# Parser class
#--------------------
class language:
    bangla={'Vowel Diacritics': ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ'],
            'Consonant Diacritics': ['র্', 'র্য', '্য', '্র', '্র্য', 'র্্র'],
            'Modifiers': ['ং', 'ঃ', 'ঁ'],
            'Connector': '্'}

class GraphemeParser(object):
    def __init__(self,language):
        '''
            initializes a grapheme parser for a given language
            args:
                language  :   a dictionary that contains list of:
                                1. Vowel Diacritics 
                                2. Consonant Diacritics
                                3. Modifiers
                                and 
                                4. Connector 
        '''
        # error check -- existace
        assert "Vowel Diacritics" in language.keys(),"Vowel Diacritics Not found"
        assert "Consonant Diacritics" in language.keys(),"Consonant Diacritics Not found"
        assert "Modifiers" in language.keys(),"Modifiers Not found"
        assert "Connector" in language.keys(),"Modifiers Not found"
        # assignment
        self.vds=language["Vowel Diacritics"]
        self.cds=language["Consonant Diacritics"]
        self.mds=language["Modifiers"]
        self.connector=language["Connector"]
        # error check -- type
        assert type(self.vds)==list,"Vowel Diacritics Is not a list"
        assert type(self.cds)==list,"Consonant Diacritics Is not a list"
        assert type(self.mds)==list,"Modifiers Is not a list"
        assert type(self.connector)==str,"Connector Is not a string"
    
    def get_root_from_decomp(self,decomp):
        '''
            creates grapheme root based list 
        '''
        add=0
        if self.connector in decomp:   
            c_idxs = [i for i, x in enumerate(decomp) if x == self.connector]
            # component wise index map    
            comps=[[cid-1,cid,cid+1] for cid in c_idxs ]
            # merge multi root
            r_decomp = []
            while len(comps)>0:
                first, *rest = comps
                first = set(first)

                lf = -1
                while len(first)>lf:
                    lf = len(first)

                    rest2 = []
                    for r in rest:
                        if len(first.intersection(set(r)))>0:
                            first |= set(r)
                        else:
                            rest2.append(r)     
                    rest = rest2

                r_decomp.append(sorted(list(first)))
                comps = rest
            # add    
            combs=[]
            for ridx in r_decomp:
                comb=''
                for i in ridx:
                    comb+=decomp[i]
                combs.append(comb)
                for i in ridx:
                    decomp[i]=comb
                    
            # new root based decomp
            new_decomp=[]
            for i in range(len(decomp)-1):
                if decomp[i] not in combs:
                    new_decomp.append(decomp[i])
                else:
                    if decomp[i]!=decomp[i+1]:
                        new_decomp.append(decomp[i])

            new_decomp.append(decomp[-1])#+add*connector
            
            return new_decomp
        else:
            return decomp

    def get_graphemes_from_decomp(self,decomp):
        '''
        create graphemes from decomp
        '''
        graphemes=[]
        idxs=[]
        for idx,d in enumerate(decomp):
            if d not in self.vds+self.mds:
                idxs.append(idx)
        idxs.append(len(decomp))
        for i in range(len(idxs)-1):
            sub=decomp[idxs[i]:idxs[i+1]]
            grapheme=''
            for s in sub:
                grapheme+=s
            graphemes.append(grapheme)
        return graphemes

    def process(self,word,return_graphemes=False):
        '''
            processes a word for creating:
            if return_graphemes=False (default):
                * components
            else:                 
                * grapheme 
        '''
        decomp=[ch for ch in word]
        decomp=self.get_root_from_decomp(decomp)
        if return_graphemes:
            return self.get_graphemes_from_decomp(decomp)
        else:
            components=[]
            for comp in decomp:
                if comp in self.vds+self.mds:
                    components.append(comp)
                else:
                    cd_val=None
                    for cd in self.cds:
                        if cd in comp:
                            comp=comp.replace(cd,"")
                            cd_val=cd
                    components.append(comp)
                    if cd_val is not None:
                        components.append(cd_val)
            return components
                            
