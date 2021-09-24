#!/bin/sh
save_path="/home/apsisdev/ansary/DATASETS/APSIS/NID/"
src_path="${save_path}source/"
cards_path="${save_path}cards/"
seg_path="${save_path}segment/"
# segment
python datagen_seg.py $src_path $cards_path $save_path
python store_seg.py $seg_path
echo succeded