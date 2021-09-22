# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
from glob import glob
from tqdm.auto import tqdm
from .utils import LOG_INFO,padToFixedHeightWidth 
import PIL
import PIL.ImageFont,PIL.Image,PIL.ImageDraw
import random
import json 
import calendar
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math
#--------------------
# source
#--------------------
class Data(object):
    def __init__(self,src_dir):
        '''
            initilizes all sources under dataset
            args:
                src_dir     :       location of source folder
            TODO:
                * ADD DOCUMENTATION IN README
                * ADD MISSING TEMPLATES
                * BACK Template 
        '''
        self.src_dir          =     src_dir
        self.res_dir          =     os.path.join(self.src_dir,"resources")
        #-----------------------
        # data folders
        #-----------------------
        LOG_INFO("Initializing Sources")
        class source:
            class styles:
                class back:
                    nid   = [img_path for img_path in  tqdm(glob(os.path.join(self.src_dir,"styles","back","nid","*.*")))]
                    smart = [img_path for img_path in  tqdm(glob(os.path.join(self.src_dir,"styles","back","smart","*.*")))]
                class front:
                    nid   = [img_path for img_path in  tqdm(glob(os.path.join(self.src_dir,"styles","front","nid","*.*")))]
                    smart = [img_path for img_path in  tqdm(glob(os.path.join(self.src_dir,"styles","front","smart","*.*")))]
            class noise:
                signs     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","signature","*.*")))]
                faces     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","faces","*.*")))]
                backs     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","background","*.*")))]
            
        #-----------------------
        # CARD CLASS
        #-----------------------
        # bboxes: [xmin,ymin,xmax,ymax]
        class card:
            class smart:
                class front:
                    template    =   os.path.join(self.res_dir,"smart_template_front.png")
                    height      =   614
                    width       =   1024
                    face        =   [57, 182, 319, 481]
                    sign        =   [57, 494, 319, 591]
                    data        =   [319, 148, 778, 591]
                    chip        =   [781, 302, 914, 399] 
                    text        ={
                                    "bn_name"     :   {"location":[327, 187, 777, 245],"font_size":48,"lang":"bn","font":"bold"},
                                    "en_name"     :   {"location":[327, 270, 777, 318],"font_size":32,"lang":"en","font":"bold"},
                                    "f_name"      :   {"location":[327, 342, 777, 403],"font_size":48,"lang":"bn","font":"reg"},
                                    "m_name"      :   {"location":[327, 417, 777, 485],"font_size":48,"lang":"bn","font":"reg"},
                                    "dob"         :   {"location":[480, 495, 777, 550],"font_size":38,"lang":"en","font":"reg"},
                                    "nid"         :   {"location":[480, 550, 777, 590],"font_size":42,"lang":"en","font":"bold"}
                                }
                class back:
                    template    =   os.path.join(self.res_dir,"smart_template_back.png")
            class nid:
                class front:
                    template    =   os.path.join(self.res_dir,"nid_template_front.png")
                    height      =   614
                    width       =   1024
                    face        =   [26, 219, 252, 451]
                    sign        =   [26, 461, 252, 574]
                    data        =   [259, 198, 1011, 600]
                    text        =   {
                                        "bn_name"     :   {"location":[410, 200, 1011, 280],"font_size":56,"lang":"bn","font":"bold"},
                                        "en_name"     :   {"location":[410, 280, 1011, 331],"font_size":36,"lang":"en","font":"bold"},
                                        "f_name"      :   {"location":[410, 331, 1011, 395],"font_size":50,"lang":"bn","font":"reg"},
                                        "m_name"      :   {"location":[410, 395, 1011, 461],"font_size":50,"lang":"bn","font":"reg"},
                                        "dob"         :   {"location":[545, 461, 1011, 520],"font_size":42,"lang":"en","font":"reg"},
                                        "nid"         :   {"location":[455, 520, 1011, 600],"font_size":60,"lang":"en","font":"bold"}
                                    }
                class back:
                    template    =   os.path.join(self.res_dir,"nid_template_back.png")

        class config:
            max_rotation  = 15
            max_warp_perc = 10 
            max_pad_perc  = 50
            noise_weights = [0.7,0.3]
            blur_weights  = [0.5,0.5]
            class bangla_name:
                max_len = 20
                puncts  = [',','.','-','(',')']
                mods    = ["মোঃ ","মোছাঃ "]
                sub_len = 3
            class english_name:
                max_len = 20
                puncts  = [',','.','-','(',')']
                mods    = ["MD. ","MRS. "]
                sub_len = 3
        
        self.source =   source
        self.card   =   card 
        self.config =   config    
        # extend text
        ## smart card font
        self.initTextFonts(self.card.smart.front.text)
        ## nid card font
        self.initTextFonts(self.card.nid.front.text)
        
        #----------------------------
        # dictionary json
        #----------------------------
        with open(os.path.join(self.res_dir,"dict.json")) as f:
            json_data       =   json.load(f)
            lang_dict       =   json_data["language"]
            self.bangla     =   lang_dict["bangla"]
            self.english    =   lang_dict["english"]
            self.vocab      =   json_data["vocab"]

        

    def __getDataFont(self,attr):
        '''
            different size font initialization
        '''
        if attr["lang"]=="bn":
            font_path=os.path.join(self.res_dir,f"bangla_{attr['font']}.ttf")
        else:
            font_path=os.path.join(self.res_dir,f"english_{attr['font']}.ttf")
        font = PIL.ImageFont.truetype(font_path, size=attr["font_size"])
        return font
    
    def initTextFonts(self,text):
        '''
            initializes text fonts
        '''
        for name,attr in text.items():
            text[name]["font"]=self.__getDataFont(attr)

    def __createBnName(self,mod_id):
        '''
            creates bangla name
        '''
        name        =   ''
        name_len    =   self.config.bangla_name.max_len
        max_punct   =   self.config.bangla_name.sub_len
        max_space   =   self.config.bangla_name.sub_len
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            start=self.config.bangla_name.mods[mod_id]
            name+=start
            name_len-=self.config.bangla_name.sub_len  
        # use puncts
        if random.choice([1,0,0,0])==1:
            use_puncts=True
        else:
            use_puncts=False
        name_len=random.randint(1,name_len)
        # string construction
        for curr_len in range(name_len):
            name+=random.choice(self.bangla["graphemes"])
            if use_puncts and max_punct>0:
                if random.choice([1,0])==1:
                    name+=random.choice(self.config.bangla_name.puncts)
                    max_punct-=1
            if (curr_len % self.config.bangla_name.sub_len) >0 and curr_len > self.config.english_name.sub_len and max_space >0:
                if random.choice([1,0])==1:
                    name+=' '
                    max_space-=1

        if name[-1]==' ':
            name=name[:-1]
        return name

    def __createEnName(self,mod_id,type):
        '''
            creates English name
        '''
        name        =   ''
        name_len    =   self.config.english_name.max_len
        max_punct   =   self.config.english_name.sub_len
        max_space   =   self.config.english_name.sub_len
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            start=self.config.english_name.mods[mod_id]
            name+=start
            name_len-=self.config.english_name.sub_len  
        # use puncts
        if random.choice([1,0])==1:
            use_puncts=True
        else:
            use_puncts=False
        name_len=random.randint(1,name_len)
        # determine case
        if type=="smart":
            use_full_upper_case=True
        else:
            if random.choice([1,0,0,0])==1:
                use_full_upper_case=True
            else:
                use_full_upper_case=False

        # string construction
        for curr_len in range(name_len):
            if use_full_upper_case:
                name+=random.choice(self.english["uppercase"])
            else:
                name+=random.choice(self.english["lowercase"]+self.english["uppercase"])
            
            if use_puncts and max_punct>0:
                if random.choice([1,0])==1:
                    name+=random.choice(self.config.english_name.puncts)
                    max_punct-=1
            if (curr_len % self.config.english_name.sub_len) >0 and curr_len > self.config.english_name.sub_len and max_space >0:
                if random.choice([1,0])==1:
                    name+=' '
                    max_space-=1
        
        if name[-1]==' ':
            name=name[:-1]
        return name
    
    def __getNumber(self,len):
        num=''
        for _ in range(len):
            num+=random.choice(self.english["numbers"])
        return num 
    def __createDOB(self,type):
        '''
            creates a date of birth
        '''
        months=list(calendar.month_abbr)[1:]
        start=self.__getNumber(2)
        if type=="smart":
            if start[0]=='0':
                start=start[1:]

        return start+' '+random.choice(months)+' '+self.__getNumber(4) 

    def __createNID(self,type):
        if type=="smart":
            return self.__getNumber(3)+' '+self.__getNumber(3)+' '+self.__getNumber(4)
        else:
            return self.__getNumber(random.randint(10,15))

    def __createTextCardFront(self,type):
        return {"bn_name":self.__createBnName(mod_id=None),
                "en_name":self.__createEnName(mod_id=None,type=type),
                "f_name" :self.__createBnName(mod_id=0),
                "m_name" :self.__createBnName(mod_id=1),
                "dob"    :self.__createDOB(type),
                "nid"    :self.__createNID(type)}

   
    
    def createCardFront(self,type,depth_color=50):
        '''
            creates an image of card front side data
        '''
        if type=="smart":
            card_front=self.card.smart.front
            info_color=(depth_color,depth_color,depth_color)
        else:
            card_front=self.card.nid.front
            info_color=(0,0,255)
        template =cv2.imread(card_front.template)
        # fill signs and images
        sign=cv2.imread(random.choice(self.source.noise.signs),0)
        face=cv2.imread(random.choice(self.source.noise.faces))
        # place face
        x1,y1,x2,y2=card_front.face
        h=y2-y1
        w=x2-x1
        template[y1:y2,x1:x2]=cv2.resize(face,(w,h))
        # place sign
        x1,y1,x2,y2=card_front.sign
        h=y2-y1
        w=x2-x1
        mask=np.ones(template.shape[:-1])*255
        mask[y1:y2,x1:x2]=cv2.resize(sign,(w,h),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
        mask[mask!=0]=255
        template[mask==0]=(0,0,0)
        # text data
        info_keys=["nid","dob"]
        
        text=self.__createTextCardFront(type)
        h_t,w_t,d=template.shape
        for k,v in text.items():
            # res
            font=card_front.text[k]["font"]
            mask=np.zeros((h_t,w_t))
            # height width
            x1,y1,x2,y2=card_front.text[k]["location"]
            width_loc=x2-x1
            height_loc=y2-y1
            (width,height), (offset_x, offset_y) = font.font.getsize(v)
            # data
            image   =   PIL.Image.new(mode='L', size=(width+offset_x,height+offset_y))
            draw    =   PIL.ImageDraw.Draw(image)
            draw.text(xy=(0,0),text=v, fill=1, font=font)
            image   =   np.array(image)
            image=padToFixedHeightWidth(image,height_loc,width_loc)
            mask[y1:y2,x1:x2]=image
            if k in info_keys:
                template[mask>0]=info_color
            else:
                template[mask>0]=(depth_color,depth_color,depth_color)
            
        return template,text
    
    def backgroundGenerator(self,dim=(1024,1024)):
        '''
        generates random background
        args:
            ds   : dataset object
            dim  : the dimension for background
        '''
        # collect image paths
        _paths=self.source.noise.backs
        while True:
            _type=random.choice(["single","double","comb"])
            if _type=="single":
                img=cv2.imread(random.choice(_paths))
                yield img
            elif _type=="double":
                imgs=[]
                img_paths= random.sample(_paths, 2)
                for img_path in img_paths:
                    img=cv2.imread(img_path)
                    h,w,d=img.shape
                    img=cv2.resize(img,dim)
                    imgs.append(img)
                # randomly concat
                img=np.concatenate(imgs,axis=random.choice([0,1]))
                img=cv2.resize(img,(w,h))
                yield img
            else:
                imgs=[]
                img_paths= random.sample(_paths, 4)
                for img_path in img_paths:
                    img=cv2.imread(img_path)
                    h,w,d=img.shape
                    img=cv2.resize(img,dim)
                    imgs.append(img)
                seg1=imgs[:2]
                seg2=imgs[2:]
                seg1=np.concatenate(seg1,axis=0)
                seg2=np.concatenate(seg2,axis=0)
                img=np.concatenate([seg1,seg2],axis=1)
                img=cv2.resize(img,(w,h))
                yield img