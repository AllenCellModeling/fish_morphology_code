#!/bin/bash

python merge_cellprofiler_output.py \
    "/allen/aics/gene-editing/FISH/2019/chaos/data/cp_testing/nuclei_filtering_by_centroid3" \
    "napari_" \
    "/allen/aics/gene-editing/FISH/2019/chaos/data/cp_testing/nuclei_filtering_by_centroid3/test_merged.csv"
