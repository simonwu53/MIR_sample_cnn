#!/usr/bin/env bash

python main.py \
  --mode test \
  --p_out ./eval \
  --loss bce \
  --p_data ./dataset/processed \
  --data_normalization \
  --batch_size 23 \
  --n_workers 4 \
  --checkpoint out/3^9-model-adamw-norm/best@epoch-027-loss-0.142773.tar \
  --tensorboard_exp_name adamw-norm
