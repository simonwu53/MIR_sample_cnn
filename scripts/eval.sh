#!/usr/bin/env bash

python main.py \
  --mode eval \
  --audio_file dataset/raw/f/american_baroque-the_four_seasons_by_vivaldi-02-concerto_no_1_in_d_major_rv_269_spring__largo-88-117.mp3 \
  --checkpoint out/3^9-model-adamw-norm/best@epoch-027-loss-0.142773.tar \
  --eval_threshold 0.3
