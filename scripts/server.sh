#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
#save_path="/media/ansary/DriveData/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
seg_path="${save_path}segment/"
rec_path="${save_path}recog/"
det_path="${save_path}detect/"
clean_rec_path="${save_path}recog/clean/"
noisy_rec_path="${save_path}recog/noisy/"
#------------------------------------------card------------------------------------------------------
#python datagen_card.py $src_path $save_path --num_data 10
#----------------------------------------------------------------------------------------------------
#------------------------------------------det------------------------------------------------------
python datagen_det.py $card_path $save_path 
python store_det.py $det_path
#---------------------------------------------------------------------------------------------------
#------------------------------------------seg------------------------------------------------------
python datagen_seg.py $src_path $card_path $save_path 
python store_seg.py $seg_path
#---------------------------------------------------------------------------------------------------
#------------------------------------------rec------------------------------------------------------
python datagen_rec.py $src_path $card_path $save_path --add_noise False
python process_rec.py $clean_rec_path
python store_rec.py $clean_rec_path
#---------------------------------------------------------------------------------------------------
python datagen_rec.py $src_path $card_path $save_path --add_noise True
python process_rec.py $noisy_rec_path
python store_rec.py $noisy_rec_path
#---------------------------------------------------------------------------------------------------
echo succeded