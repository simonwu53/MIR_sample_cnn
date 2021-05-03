#!/usr/bin/env bash

python main.py \
  --mode train \
  --max_train 5 \
  --max_epoch 100 \
  --early_stop_patience 5 \
  --p_out ./out \
