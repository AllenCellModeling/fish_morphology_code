#!/bin/bash

# Run cellprofiler pipeline in headless mode

cellprofiler \
    -p cp_3i_image_processing.cppipe \
    --run-headless \
    --data-file image_set_template.csv \
    -o ~/cp_out \
    -L 10
