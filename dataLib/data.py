# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
from glob import glob
from unicodedata import name
from tqdm.auto import tqdm
from .utils import LOG_INFO,padToFixedHeightWidth,GraphemeParser 
import PIL
import PIL.ImageFont,PIL.Image,PIL.ImageDraw
import random
import json 
import calendar
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math
GP=GraphemeParser()
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
                                    "dob"         :   {"location":[480, 490, 777, 550],"font_size":38,"lang":"en","font":"reg"},
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
                                    "bn_name"     :   {"location":[410, 210, 1011, 280],"font_size":56,"lang":"bn","font":"bold"},
                                    "en_name"     :   {"location":[410, 282, 1011, 322],"font_size":36,"lang":"en","font":"bold"},
                                    "f_name"      :   {"location":[410, 338, 1011, 390],"font_size":52,"lang":"bn","font":"reg"},
                                    "m_name"      :   {"location":[410, 400, 1011, 460],"font_size":52,"lang":"bn","font":"reg"},
                                    "dob"         :   {"location":[545, 455, 1011, 515],"font_size":42,"lang":"en","font":"reg"},
                                    "nid"         :   {"location":[455, 515, 1011, 600],"font_size":60,"lang":"en","font":"bold"}
                                    }
                class back:
                    template    =   os.path.join(self.res_dir,"nid_template_back.png")

        class config:
            max_rotation  = 5
            max_warp_perc = 10 
            max_pad_perc  = 20
            noise_weights = [0.7,0.3]
            blur_weights  = [0.5,0.5]
            use_scope_rotation=False
        class name:
            max_words  = 5
            min_words  = 2
            min_len    = 1
            max_len    = 8
            total      = 60
            seps        = ["",".",","]
            sep_weights = [0.6,0.2,0.2]
            frag_weights= [0.8,0.2]


        self.source =   source
        self.card   =   card 
        self.config =   config
        self.name   =   name     
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

    #----------------------------
    # text construction
    #----------------------------
    def __createNameData(self,vocabs,language):
        num_word=random.randint(self.name.min_words,self.name.max_words)
        words=[]
        # fragment
        if random.choices(population=[0,1],weights=self.name.frag_weights,k=1)[0]==1:
            use_fragment=True
        else:
            use_fragment=False
        
        for _ in range(num_word):
            num_vocab=random.randint(self.name.min_len,self.name.max_len)
            word="".join([random.choice(vocabs) for _ in range(num_vocab)])
            
            # check invalid bangla starting:
            if language=="bangla" and word[0] in ["ঁ","ং","ঃ"]:
                if num_vocab>1:
                    word=word[1:]
                elif num_vocab==1:
                    word=''
            # blank word filter
            if word=='':
                continue

            # handling . and ,  (seps)       
            if num_vocab==1 and not use_fragment:
                if language=="english":word+="."
                else:word+=random.choice([".",","])
            elif num_vocab==2 and language=="bangla" and not use_fragment: 
                word+=random.choices(population=self.name.seps,weights=self.name.sep_weights,k=1)[0]
            words.append(word)
        # check length
        name=" ".join(words)
        while len(name)>self.name.total:
            words=words[:-1]
            name=" ".join(words)
        
        if use_fragment:
            # bracket last word
            if random.choices(population=[0,1],weights=[0.5,0.5],k=1)[0]==1:
                words[-1]="("+words[-1]+")"
            # hyphenate 3 words
            else:
                if len(words)>3:
                    connect=[]
                    for _ in range(3):
                        idx=random.choice([i for i in range(len(words))])
                        connect.append(words[idx])
                        words[idx]=None
                        words=[word for word in words if word is not None]
                    name="-".join(connect)
                    return name
        name=" ".join(words)
        return name

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
        mods    = ["মোঃ ","মোছাঃ "]
        name        =   ''
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            name+=mods[mod_id]
        #
        name+=self.__createNameData(self.bangla["graphemes"],"bangla")
        return name

    def __createEnName(self,mod_id,type):
        '''
            creates English name
        '''
        mods    = ["MD. ","MRS. "]
        name        =   ''
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            name+=mods[mod_id]
              
        # determine case
        if type=="smart":
            vocabs=self.english["uppercase"]
        else:
            if random.choice([1,0])==1:
                vocabs=self.english["uppercase"]
            else:
                vocabs=self.english["uppercase"]+self.english["lowercase"]

        name=self.__createNameData(vocabs,"english")
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
        template_label={}
        iden=2
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
        
        # text data processing
        text=self.__createTextCardFront(type)
        h_t,w_t,d=template.shape
        template_mask=np.zeros((h_t,w_t))
        for k,v in text.items():
            # res
            font=card_front.text[k]["font"]
            mask=np.zeros((h_t,w_t))
            # height width
            x1,y1,x2,y2=card_front.text[k]["location"]
            width_loc=x2-x1
            height_loc=y2-y1
            
            # comps
            comps=GP.word2grapheme(v)
            w_text,h_text=font.getsize(v)
            comp_str=''
            images=[]
            label={}
            for comp in comps:
                comp_str+=comp
                # data
                image   =   PIL.Image.new(mode='L', size=(w_text,h_text))
                draw    =   PIL.ImageDraw.Draw(image)
                draw.text(xy=(0,0),text=comp_str, fill=1, font=font)
                image   =   np.array(image)
                images.append(image)
                label[iden]=comp
                iden+=1
            
            img=sum(images)
            # offset
            vals=list(np.unique(img))
            vals=sorted(vals,reverse=True)
            vals=vals[:-1]
            
            image=np.zeros(img.shape)
            for lv,l in zip(vals,label.keys()):
                if l!=' ':
                    image[img==lv]=l
            
            # crop to size
            tidx    =   np.where(image>0)
            y_min,y_max,x_min,x_max = np.min(tidx[0]), np.max(tidx[0]), np.min(tidx[1]), np.max(tidx[1])
            image=image[y_min:y_max,x_min:x_max]
            # pad
            image=padToFixedHeightWidth(image,height_loc,width_loc)
            mask[y1:y2,x1:x2]=image
            template_mask[y1:y2,x1:x2]=image
            if k in info_keys:
                template[mask>0]=info_color
            else:
                template[mask>0]=(depth_color,depth_color,depth_color)
            template_label[k]=label
            
        return template,template_mask,template_label
    
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