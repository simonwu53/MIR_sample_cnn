#!/usr/bin/env bash

BASE_DIR="path/to/mtt/base"
N_WORKER=12

python src/data/preprocess.py \
 --n_worker ${N_WORKER} \
 --p_anno "${BASE_DIR}/annotations_final.csv" \
 --p_info "${BASE_DIR}/clip_info_final.csv" \
 --p_raw "${BASE_DIR}/raw" \
 --p_out "${BASE_DIR}/processed"