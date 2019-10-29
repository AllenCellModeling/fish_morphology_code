#!/usr/bin/env python


import os
import fire

from fish_morphology_code.processing.merge_features.cp_processing_utils import (
    image_processing_errors,
    merge_cellprofiler_output,
    add_sample_image_metadata,
    add_cell_structure_scores,
    cat_structure_scores,
    remove_missing_images,
    DEFAULT_CELLPROFILER_CSVS,
)


def run(
    cp_csv_dir="",
    csv_prefix="napari_",
    out_csv="./merged_features.csv",
    normalized_image_manifest="",
    fov_raw_seg_manifest="",
    fov_metadata="",
    structure_score_paths="",
):
    """
    Merge cellprofiler output csvs with image metadata and structure scores
    Args:
        cp_csv_dir (str): location of cellprofiler output csvs
        csv_prefix (str): optional prefix part of cellprofiler output csv file names
        out_csv (str): path to file where to save merged csv
        normalized_image_manifest (str): location of contrast stretched images that were cellprofiler input
        fov_raw_seg_manifest (str): location of manifest that matches raw image file names with pre-contrast stretched names
        fov_metadata (str): location of csv file with sample metadata from labkey
        structure_score_paths (str): location of csv file with manual structure scores per cell
    """

    image_csv = os.path.join(
        cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["image"]
    )

    failed_images = image_processing_errors(image_csv)

    # merge cellprofiler output csvs
    cp_feature_df = merge_cellprofiler_output(
        image_csv=image_csv,
        merged_nuclei_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["merged_nuclei"]
        ),
        flag_border_nuclei_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["flag_border_nuclei"]
        ),
        napari_cell_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["napari_cell"]
        ),
        premerge_nuclei_centroids_csv=os.path.join(
            cp_csv_dir,
            csv_prefix + DEFAULT_CELLPROFILER_CSVS["premerge_nuclei_centroids"],
        ),
        failed_images=failed_images,
    )

    # add sample and image metadata to cp feature data frame
    cp_feature_metadata_df = add_sample_image_metadata(
        cell_feature_df=cp_feature_df,
        norm_image_manifest=normalized_image_manifest,
        fov_raw_seg=fov_raw_seg_manifest,
        fov_metadata=fov_metadata,
    )

    cell_structure_scores_df = cat_structure_scores(score_files=structure_score_paths)

    # add manual structure scores to feature data frame
    feature_df = add_cell_structure_scores(
        cell_feature_df=cp_feature_metadata_df,
        structure_score_df=cell_structure_scores_df,
    )

    final_feature_df = remove_missing_images(feature_df)

    final_feature_df.to_csv(out_csv, index=False)


def main():
    fire.Fire(run)
