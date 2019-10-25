#!/usr/bin/env python

"""
Flag images with processing failures anywhere in the cellprofiler pipeline
"""

import os
import argparse

from cp_processing_utils import (
    image_processing_errors,
    merge_cellprofiler_output,
    add_sample_image_metadata,
    add_cell_structure_scores,
    cat_structure_scores,
    remove_missing_images,
    DEFAULT_CELLPROFILER_CSVS,
)

parser = argparse.ArgumentParser(
    description="Merge cellprofiler output csvs and flag images with processing failures"
)
parser.add_argument(
    "--cp_csv_dir",
    required=True,
    type=str,
    default="./",
    help="path to csv files output by cellprofiler",
)
parser.add_argument(
    "--csv_prefix",
    required=True,
    type=str,
    default="napari_",
    help="optional prefix part of name of cellprofiler output csvs",
)
parser.add_argument(
    "--out_csv", required=True, type=str, help="path to file where to save merged csv"
)
parser.add_argument(
    "--normalized_image_manifest",
    required=True,
    type=str,
    help="path to image normalization output manifest",
)
parser.add_argument(
    "--fov_raw_seg_manifest",
    required=True,
    type=str,
    help="path to manifest with raw and un-normalized processed images",
)
parser.add_argument(
    "--fov_metadata",
    required=True,
    type=str,
    help="path to sample metadata file from labkey fov table",
)
parser.add_argument(
    "--structure_score_paths",
    required=True,
    type=str,
    help="path to csv that contains locations of multiple plate structure score csvs",
)

args = parser.parse_args()

image_csv = os.path.join(
    args.cp_csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["image"]
)

failed_images = image_processing_errors(image_csv)

# merge cellprofiler output csvs
cp_feature_df = merge_cellprofiler_output(
    image_csv=image_csv,
    merged_nuclei_csv=os.path.join(
        args.cp_csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["merged_nuclei"]
    ),
    flag_border_nuclei_csv=os.path.join(
        args.cp_csv_dir,
        args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["flag_border_nuclei"],
    ),
    napari_cell_csv=os.path.join(
        args.cp_csv_dir, args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["napari_cell"]
    ),
    premerge_nuclei_centroids_csv=os.path.join(
        args.cp_csv_dir,
        args.csv_prefix + DEFAULT_CELLPROFILER_CSVS["premerge_nuclei_centroids"],
    ),
    failed_images=failed_images,
)

# add sample and image metadata to cp feature data frame
cp_feature_metadata_df = add_sample_image_metadata(
    cell_feature_df=cp_feature_df,
    norm_image_manifest=args.normalized_image_manifest,
    fov_raw_seg=args.fov_raw_seg_manifest,
    fov_metadata=args.fov_metadata,
)

cell_structure_scores_df = cat_structure_scores(score_files=args.structure_score_paths)

# add manual structure scores to feature data frame
feature_df = add_cell_structure_scores(
    cell_feature_df=cp_feature_metadata_df, structure_score_df=cell_structure_scores_df
)

final_feature_df = remove_missing_images(feature_df)

final_feature_df.to_csv(args.out_csv, index=False)
