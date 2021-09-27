#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
#save_path="/media/ansary/DriveData/Work/APSIS/datasets/NID/"
src_path="${save_path}source/"
card_path="${save_path}cards/"
seg_path="${save_path}segment/"
rec_path="${save_path}recog/"
clean_rec_path="${save_path}recog/clean/"
noisy_rec_path="${save_path}recog/noisy/"
# # card data
# python datagen_card.py $src_path $save_path --num_data 50000
# # noisy rec
# python datagen_rec.py $src_path $card_path $save_path --add_noise True
# # clean rec
# python datagen_rec.py $src_path $card_path $save_path --add_noise False
python process_rec.py $clean_rec_path
python process_rec.py $noisy_rec_path
python store_rec.py $clean_rec_path
python store_rec.py $noisy_rec_path
echo succeded