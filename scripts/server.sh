#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
#save_path="/media/ansary/DriveData/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
seg_path="${save_path}segment/"
rec_path="${save_path}recog/"
det_path="${save_path}detect/"
proc_path="${save_path}processed/"
#------------------------------------------card------------------------------------------------------
python datagen_card.py $src_path $save_path --num_data 20000
#----------------------------------------------------------------------------------------------------
#------------------------------------------rec------------------------------------------------------
python datagen_rec.py $src_path $card_path $save_path 
python process_rec.py $rec_path $save_path
python store_rec.py $proc_path
#---------------------------------------------------------------------------------------------------
#------------------------------------------det------------------------------------------------------
#python datagen_det.py $card_path $save_path 
#ython store_det.py $det_path
#---------------------------------------------------------------------------------------------------
#------------------------------------------seg------------------------------------------------------
#python datagen_seg.py $src_path $card_path $save_path 
#python store_seg.py $seg_path
#---------------------------------------------------------------------------------------------------
echo succeded