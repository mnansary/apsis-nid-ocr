#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
src_path="${save_path}source/"
card_path="${save_path}cards/"
seg_path="${save_path}segment/"
# card data
python datagen_card.py $src_path $save_path --num_data 50000
# noisy rec
python datagen_rec.py $src_path $card_path $save_path --add_noise True
# clean rec
python datagen_rec.py $src_path $card_path $save_path --add_noise False
echo succeded