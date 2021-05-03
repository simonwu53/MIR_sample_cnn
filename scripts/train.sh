#!/usr/bin/env bash

python main.py \
  --mode train \
  --p_out ./out \
  --optim_type sgd \
  --init_lr 1e-2 \
  --min_lr 0.000016 \
  --lr_decay_global 0.2 \
  --lr_decay_plateau 0.2 \
  --plateau_patience 3 \
  --lr_decay_local 0.999999 \
  --early_stop_patience 10 \
  --early_stop_delta 0 \
  --loss bce \
  --momentum 0.9 \
  --max_train 5 \
  --max_epoch 100 \
  --p_data ./dataset/processed \
  --batch_size 23 \
  --n_workers 4 \
  --use_best_for_stage \
  --tensorboard_exp_name orig
