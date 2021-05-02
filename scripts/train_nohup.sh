#!/usr/bin/env bash

LOG_FILE="logs/train.log"

nohup python main.py \
        --mode train \
        --max_train 1 \
        --max_epoch 1  >> "${LOG_FILE}" &

tail -F "${LOG_FILE}"