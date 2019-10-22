#!/usr/bin/env python

"""
Flag images with processing failures anywhere in the cellprofiler pipeline
"""

import os
import argparse

from cp_processing_utils import (
    image_processing_errors,
    mergeCellProfilerOutput,
    DEFAULT_CELLPROFILER_CSVS,
)

parser = argparse.ArgumentParser(
    description="Merge cellprofiler output csvs and flag images with processing failures"
)
parser.add_argument(
    "csv_dir", type=str, default="./", help="path to csv files output by cellprofiler"
)
parser.add_argument(
    "csv_prefix",
    type=str,
    default="napari_",
    help="optional prefix part of name of cellprofiler output csvs",
)
parser.add_argument("out_csv", type=str, help="path to file where to save merged csv")

args = parser.parse_args()

image_csv = os.path.join(
    args.csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["image"]
)

failed_images = image_processing_errors(image_csv)

merged_df = mergeCellProfilerOutput(
    image_csv,
    os.path.join(
        args.csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["merged_nuclei"]
    ),
    os.path.join(
        args.csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["flag_border_nuclei"]
    ),
    os.path.join(
        args.csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["napari_cell"]
    ),
    os.path.join(
        args.csv_dir,
        args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["premerge_nuclei_centroids"],
    ),
)

merged_df.to_csv(args.out_csv)
