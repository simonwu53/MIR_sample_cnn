#!/usr/bin/env bash

python main.py \
  --mode train \
  --device cuda:0 \
  --model samplecnn \
  --m 3 \
  --n 9 \
  --module_filters 128 128 128 256 256 256 256 256 256 512 512 \
  --p_out ./out \
  --optim_type adamw \
  --lr 1e-2 \
  --min_lr 0.000016 \
  --lr_decay_plateau 0.2 \
  --plateau_patience 5 \
  --early_stop_patience 15 \
  --loss bce \
  --max_epoch 50 \
  --p_data ./dataset/processed \
  --data_normalization \
  --batch_size 23 \
  --n_workers 4 \
  --tensorboard_exp_name adamw-norm
