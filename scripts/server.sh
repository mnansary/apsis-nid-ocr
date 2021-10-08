#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
#save_path="/media/ansary/DriveData/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
class_path="${save_path}classification/"
cseg_path="${save_path}segment/"
rec_path="${save_path}recog/"
det_path="${save_path}detect/"
proc_path="${save_path}processed/"
#------------------------------------------card------------------------------------------------------
#python datagen_card.py $src_path $save_path --num_data 50000
#----------------------------------------------------------------------------------------------------
#------------------------------------------class------------------------------------------------------
#python datagen_class.py $src_path $card_path $save_path 
#python store_class.py $class_path
#---------------------------------------------------------------------------------------------------
#------------------------------------------cseg------------------------------------------------------
#python datagen_cseg.py $src_path $card_path $save_path 
#python store_cseg.py $cseg_path
#---------------------------------------------------------------------------------------------------

#------------------------------------------det------------------------------------------------------
#python datagen_det.py $card_path $save_path 
#python store_det.py $det_path
#---------------------------------------------------------------------------------------------------

#------------------------------------------rec------------------------------------------------------
#python datagen_rec.py $src_path $card_path $save_path 
python process_rec.py $rec_path $save_path
python store_rec.py $proc_path
#---------------------------------------------------------------------------------------------------

echo succeded