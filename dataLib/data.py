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
from .utils import LOG_INFO 
import PIL
import PIL.ImageFont
import random
import json 
import calendar
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
            class fonts:
                bangla    = [font_path for font_path in  tqdm(glob(os.path.join(self.src_dir,"fonts","bangla","*.*")))]
                english   = [font_path for font_path in  tqdm(glob(os.path.join(self.src_dir,"fonts","english","*.*")))]
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
                                    "bn_name"     :   {"location":[319, 192, 778, 237],"font_size":42,"lang":"bn"},
                                    "en_name"     :   {"location":[319, 267, 778, 311],"font_size":42,"lang":"en"},
                                    "f_name"      :   {"location":[319, 337, 778, 386],"font_size":42,"lang":"bn"},
                                    "m_name"      :   {"location":[319, 413, 778, 466],"font_size":42,"lang":"bn"},
                                    "dob"         :   {"location":[456, 466, 778, 525],"font_size":42,"lang":"en"},
                                    "nid"         :   {"location":[456, 529, 778, 579],"font_size":48,"lang":"en"}
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
                                        "bn_name"     :   {"location":[388, 206, 1011, 266],"font_size":48,"lang":"bn"},
                                        "en_name"     :   {"location":[388, 272, 1011, 326],"font_size":48,"lang":"en"},
                                        "f_name"      :   {"location":[388, 331, 1011, 396],"font_size":48,"lang":"bn"},
                                        "m_name"      :   {"location":[388, 399, 1011, 456],"font_size":48,"lang":"bn"},
                                        "dob"         :   {"location":[531, 461, 1011, 512],"font_size":48,"lang":"en"},
                                        "nid"         :   {"location":[451, 528, 1011, 600],"font_size":64,"lang":"en"}
                                    }
                class back:
                    template    =   os.path.join(self.res_dir,"nid_template_back.png")

        class config:
            class bangla_name:
                max_len = 15
                puncts  = [',','.','-','(',')']
                mods    = ["মোঃ ","মোছাঃ "]
                sub_len = 3
            class english_name:
                max_len = 15
                puncts  = [',','.','-','(',')']
                mods    = ["MD. ","MRS. "]
                sub_len = 3
        
        self.source =   source
        self.card   =   card 
        self.config =   config    
        '''
        # extend text
        ## smart card font
        self.initTextFonts(self.card.smart.front.text)
        ## nid card font
        self.initTextFonts(self.card.nid.front.text)
        '''
        
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
            font_path=random.choice(self.source.fonts.bangla)
        else:
            font_path=random.choice(self.source.fonts.english)
        return PIL.ImageFont.truetype(font_path, size=attr["font_size"])
    
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

    def __createEnName(self,mod_id):
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
    def __createDOB(self):
        '''
            creates a date of birth
        '''
        months=list(calendar.month_abbr)[1:]
        return self.__getNumber(2)+' '+random.choice(months)+' '+self.__getNumber(4) 

    def __createNID(self,type):
        nid=''
        if type=="smart":
            return self.__getNumber(3)+' '+self.__getNumber(3)+' '+self.__getNumber(4)
        else:
            return self.__getNumber(random.randint(10,15))

    def createCardData(self,type):
        return {"bn_name":self.__createBnName(mod_id=None),
                "en_name":self.__createEnName(mod_id=None),
                "f_name" :self.__createBnName(mod_id=0),
                "m_name" :self.__createBnName(mod_id=1),
                "dob"    :self.__createDOB(),
                "nid"    :self.__createNID(type)}