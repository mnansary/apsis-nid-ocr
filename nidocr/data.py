#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
class card:
    height   =  614
    width    =  1024
    class nid:
        class front:
            face        =   [26, 219, 252, 451]
            sign        =   [26, 461, 252, 574]    
            box_dict = {
                    "Bangla Name"    :  [405, 195, 1011, 275],
                    "English Name"   :  [405, 275, 1011, 326],
                    "Fathers Name"   :  [405, 326, 1011, 395],
                    "Mothers Name"   :  [405, 395, 1011, 456],
                    "Date of Birth"  :  [540, 456, 1011, 515],
                    "ID No."         :  [450, 515, 1011, 595]
                    }
    class smart:
        class front:
            face        =   [57, 182, 319, 481]
            sign        =   [57, 494, 319, 591]        
            box_dict={
                    "Bangla Name"    :   [322, 190, 770, 235],
                    "English Name"   :   [322, 265, 770, 305],
                    "Fathers Name"   :   [322, 335, 770, 390],
                    "Mothers Name"   :   [322, 415, 770, 480],
                    "Date of Birth"  :   [465, 480, 770, 525],
                    "ID No."         :   [465, 530, 770, 585]
                    }
