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
#---------------------------------------------------------------
def threshold_image(img):
    '''
        threshold an image
    '''
    assert len(img.shape)==3
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

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
                min_ops=2,
                max_ops=6,
                blur_kernel_size_max=8,
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
        self.min_ops                =   min_ops
        self.max_ops                =   max_ops
        self.ops                    =   [self.__blur,
                                         self.__gaussBlur,
                                         self.__medianBlur,
                                         self.__biFilter,
                                         self.__gaussNoise,
                                         self.__addBrightness]

    def __initParams(self):
        self.blur_kernel_size=random.randrange(self.blur_kernel_size_min,
                                               self.blur_kernel_size_max, 
                                               2)
        self.bi_filter_dim   =random.randrange(self.bi_filter_dim_min,
                                               self.bi_filter_dim_max, 
                                               2)
        self.bi_filter_sigma =random.randint(self.bi_filter_sigma_min,
                                             self.bi_filter_sigma_max)
        self.num_ops         =random.randint(self.min_ops,self.max_ops)

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

    def __gaussNoise(self,img):
        h,w,d=img.shape
        noise=np.random.normal(0,1,img.size)
        noise=noise.reshape(h,w,d)
        noise=noise.astype("uint8")
        return cv2.add(img,noise)
    
    def __addBrightness(self,image):    
        ## Conversion to HLS
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
        for _ in range(self.num_ops):
            img=img.astype("uint8")
            img=random.choice(self.ops)(img)
        return img
#--------------------
# Parser class
#--------------------
class GraphemeParser():
    def __init__(self):
        self.vds    =['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
        self.cds    =['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
        self.roots  =['ং','ঃ','অ','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ক','ক্ক','ক্ট','ক্ত','ক্ল','ক্ষ','ক্ষ্ণ',
                    'ক্ষ্ম','ক্স','খ','গ','গ্ধ','গ্ন','গ্ব','গ্ম','গ্ল','ঘ','ঘ্ন','ঙ','ঙ্ক','ঙ্ক্ত','ঙ্ক্ষ','ঙ্খ','ঙ্গ','ঙ্ঘ','চ','চ্চ',
                    'চ্ছ','চ্ছ্ব','ছ','জ','জ্জ','জ্জ্ব','জ্ঞ','জ্ব','ঝ','ঞ','ঞ্চ','ঞ্ছ','ঞ্জ','ট','ট্ট','ঠ','ড','ড্ড','ঢ','ণ',
                    'ণ্ট','ণ্ঠ','ণ্ড','ণ্ণ','ত','ত্ত','ত্ত্ব','ত্থ','ত্ন','ত্ব','ত্ম','থ','দ','দ্ঘ','দ্দ','দ্ধ','দ্ব','দ্ভ','দ্ম','ধ',
                    'ধ্ব','ন','ন্জ','ন্ট','ন্ঠ','ন্ড','ন্ত','ন্ত্ব','ন্থ','ন্দ','ন্দ্ব','ন্ধ','ন্ন','ন্ব','ন্ম','ন্স','প','প্ট','প্ত','প্ন',
                    'প্প','প্ল','প্স','ফ','ফ্ট','ফ্ফ','ফ্ল','ব','ব্জ','ব্দ','ব্ধ','ব্ব','ব্ল','ভ','ভ্ল','ম','ম্ন','ম্প','ম্ব','ম্ভ',
                    'ম্ম','ম্ল','য','র','ল','ল্ক','ল্গ','ল্ট','ল্ড','ল্প','ল্ব','ল্ম','ল্ল','শ','শ্চ','শ্ন','শ্ব','শ্ম','শ্ল','ষ',
                    'ষ্ক','ষ্ট','ষ্ঠ','ষ্ণ','ষ্প','ষ্ফ','ষ্ম','স','স্ক','স্ট','স্ত','স্থ','স্ন','স্প','স্ফ','স্ব','স্ম','স্ল','স্স','হ',
                    'হ্ন','হ্ব','হ্ম','হ্ল','ৎ','ড়','ঢ়','য়']

        self.lowercase= ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        self.numbers= ["0","1","2","3","4","5","6","7","8","9","০","১","২","৩","৪","৫","৬","৭","৮","৯"]
        self.punctuations=[ "!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",
                            ">","?","@","[","\\","]","^","_","`","{","|","}","~","।"]
        self.uppercase=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

        self.ignore=self.punctuations+self.numbers+self.lowercase+self.uppercase+[" "]

    def word2grapheme(self,word):
        graphemes = []
        grapheme = ''
        i = 0
        while i < len(word):
            if word[i] in self.ignore:
                graphemes.append(word[i])
            else:
                grapheme += (word[i])
                # print(word[i], grapheme, graphemes)
                # deciding if the grapheme has ended
                if word[i] in ['\u200d', '্']:
                    # these denote the grapheme is contnuing
                    pass
                elif word[i] == 'ঁ':  
                    # 'ঁ' always stays at the end
                    graphemes.append(grapheme)
                    grapheme = ''
                elif word[i] in list(self.roots) + ['়']:
                    # root is generally followed by the diacritics
                    # if there are trailing diacritics, don't end it
                    if i + 1 == len(word):
                        graphemes.append(grapheme)
                    elif word[i + 1] not in ['্', '\u200d', 'ঁ', '়'] + list(self.vds):
                        # if there are no trailing diacritics end it
                        graphemes.append(grapheme)
                        grapheme = ''

                elif word[i] in self.vds:
                    # if the current character is a vowel diacritic
                    # end it if there's no trailing 'ঁ' + diacritics
                    # Note: vowel diacritics are always placed after consonants
                    if i + 1 == len(word):
                        graphemes.append(grapheme)
                    elif word[i + 1] not in ['ঁ'] + list(self.vds):
                        graphemes.append(grapheme)
                        grapheme = ''

            i = i + 1
            # Note: df_cd's are constructed by df_root + '্'
            # so, df_cd is not used in the code

        return graphemes
    