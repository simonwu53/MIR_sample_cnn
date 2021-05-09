#!/usr/bin/env bash

BASE_DIR="path/to/mtt/base"

python src/data/audio.py \
 --p_data "${BASE_DIR}/processed" \
 --split "train" \
