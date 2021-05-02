#!/usr/bin/env bash

python main.py \
  --mode train \
  --max_train 1 \
  --max_epoch 1 \
  --early_stop_patience 5
