# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from .utils import *
#--------------------
# mask data
#--------------------
class nid:
    front = {
                1:[26, 219, 252, 451],
                2:[26, 461, 252, 574],
                3:[420, 230, 1000, 260],
                4:[420, 290, 1000, 320],
                5:[420, 350, 1000, 380],
                6:[420, 410, 1000, 440],
                7:[550, 470, 1000, 500],
                8:[500, 550, 1000, 580]
            }
class smart:
    front={
            1:[57, 182, 319, 481],
            2:[57, 494, 319, 591],
            3:[325, 200, 750, 220],
            4:[325, 280, 750, 300],
            5:[325, 350, 750, 370],
            6:[325, 440, 750, 460],
            7:[465, 510, 750, 530],
            8:[465, 560, 750, 580]
            }
#--------------------
# augment data
#--------------------
def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat,rotation_mat

def get_image_coords(curr_coord,M):
    '''
        returns rotated co-ords
        args:
            curr_coord  : list of co-ords
            M           : rotation matrix
    '''
    curr_coord=np.float32(curr_coord)
    # co-ord change
    new_coord=[]
    curr_coord=np.concatenate([curr_coord,np.ones((4,1))],axis=1)
    for c in curr_coord:
        dot=np.dot(M,c)
        new_coord.append([int(i) for i in dot])
    return new_coord

    
def get_warped_image(img,mask,src,config,warp_vec):
    '''
        returns warped image and new coords
        args:
            img      : image to warp
            mask     : mask for augmentation
            src      : list of current coords
            config   : warping config
            warp_vec : which vector to warp  
    '''
    height,width,_=img.shape
 
    # construct dict warp
    x1,y1=src[0]
    x2,y2=src[1]
    x3,y3=src[2]
    x4,y4=src[3]
    # warping calculation
    xwarp=random.randint(0,config.max_warp_perc)/100
    ywarp=random.randint(0,config.max_warp_perc)/100
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(src),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height))
    mask= cv2.warpPerspective(mask, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,mask,dst
#---------------------------------------------------------------------------------------
def augment_img_base(img_path,config):
    '''
        augments a base image:
        args:
            img_path   : path of the image to augment
            config     : augmentation config
                         * max_rotation
                         * max_warping_perc
        return: 
            augmented image,augment_mask,augmented_location
    '''
    if "nid" in img_path:
        card=nid
    else:
        card=smart

    img=cv2.imread(img_path)
    height,width,d=img.shape
    warp_types=["p1","p2","p3","p4"]
    
    mask_path=img_path.replace("images","masks")
    mask=cv2.imread(mask_path,0)
    mask[mask>0]=1
    mask[mask<1]=2
    for k,v in card.front.items():
        if k in [1,2]:
            x_min,y_min,x_max,y_max=v
            mask[y_min:y_max,x_min:x_max]=k+2
    mask=mask*25
    mask=mask.astype("uint8") 
        
    curr_coord=[[0,0], 
                [width-1,0], 
                [width-1,height-1], 
                [0,height-1]]
    
    # warp
    for i in range(2):
        if i==0:
            idxs=[0,2]
        else:
            idxs=[1,3]
        if random_exec():    
            idx=random.choice(idxs)
            img,mask,curr_coord=get_warped_image(img,mask,curr_coord,config,warp_types[idx])

    if random_exec(): 
        # plane rotation
        angle=random.randint(-config.max_rotation,config.max_rotation)
        img,M =rotate_image(img,angle)
        mask,_=rotate_image(mask,angle)
        curr_coord=get_image_coords(curr_coord,M)
        
    # scope rotation
    if config.use_scope_rotation:
        if random_exec():
            flip_op=random.choice([-90,180,90])                  
            img,M=rotate_image(img,flip_op)
            mask,_=rotate_image(mask,flip_op)
            curr_coord=get_image_coords(curr_coord,M)
            
   
    return img,mask,curr_coord

#---------------------------------------------------------------------------------------
def pad_image_mask(img,mask,coord,config):
    '''
        pads data 
    '''
    h,w,d=img.shape
    coord=np.array(coord)
    # change vars
    w_pad_left=int(w*( random.randint(2,config.max_pad_perc) /100))
    h_pad_top =int(h*( random.randint(2,config.max_pad_perc) /100))
    # correct co-ordinates lr
    coord[:,0]+=w_pad_left
    # image left right
    left_pad=np.zeros((h,w_pad_left,d))
    w_pad_right=int(w*( random.randint(2,config.max_pad_perc) /100))
    right_pad=np.zeros((h,w_pad_right,d))
    img=np.concatenate([left_pad,img,right_pad],axis=1)
    mask=np.concatenate([cv2.cvtColor(left_pad.astype("uint8"),cv2.COLOR_BGR2GRAY) ,
                        mask,
                        cv2.cvtColor(right_pad.astype("uint8"),cv2.COLOR_BGR2GRAY)],axis=1)
    # correct co-ordinates lr
    coord[:,1]+=h_pad_top
    # image top bottom
    h,w,d=img.shape
    top_pad=np.zeros((h_pad_top,w,d))
    h_pad_bot=int(h*( random.randint(2,config.max_pad_perc) /100))
    bot_pad=np.zeros((h_pad_bot,w,d))
    img=np.concatenate([top_pad,img,bot_pad],axis=0)
    mask=np.concatenate([cv2.cvtColor(top_pad.astype("uint8"),cv2.COLOR_BGR2GRAY),
                        mask,
                        cv2.cvtColor(bot_pad.astype("uint8"),cv2.COLOR_BGR2GRAY)],axis=0)
    return img,mask,coord
    

def render_data(backgen,img_path,config):
    '''
        renders proper data for modeling
        args: 
            backgen : generates image background
            img_path: image to render
            config  : config for augmentation
    '''
    
    # base augment
    img,mask,coord=augment_img_base(img_path,config)    
        
    # background
    if random_exec(weights=[0.95,0.05],match=0):
        # pad
        img,mask,coord=pad_image_mask(img,mask,coord,config)
        back=next(backgen)
        h,w,d=img.shape
        back=cv2.resize(back,(w,h))
    else:
        back=np.copy(img)
        coord=np.array(coord)

    back[mask>0]=img[mask>0]
    mask=mask/25
    mask=mask.astype("uint8")
    return back,mask,coord