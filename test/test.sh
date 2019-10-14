#! /usr/bin/env bash

OUTPUT_DIR="output_data"

rm -rf ${OUTPUT_DIR} 
python ../code/stretch.py input_data/input_tiffs.csv input_data/channel_defs.json --out_dir=${OUTPUT_DIR}

