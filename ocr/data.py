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
                    "Fathers Name"   :  [405, 326, 1011, 390],
                    "Mothers Name"   :  [405, 390, 1011, 456],
                    "Date of Birth"  :  [540, 456, 1011, 515],
                    "ID No."         :  [450, 515, 1011, 595]
                    }
    class smart:
        class front:
            face        =   [57, 182, 319, 481]
            sign        =   [57, 494, 319, 591]        
            box_dict={
                    "Bangla Name"    :   [322, 175, 777, 265],
                    "English Name"   :   [322, 265, 777, 332],
                    "Fathers Name"   :   [322, 332, 777, 408],
                    "Mothers Name"   :   [322, 408, 777, 490],
                    "Date of Birth"  :   [475, 490, 777, 545],
                    "ID No."         :   [475, 545, 777, 585]
                    }    
            