#!/usr/bin/env python


import os
import fire

from fish_morphology_code.processing.merge_features.cp_processing_utils import (
    image_processing_errors,
    merge_cellprofiler_output,
    add_sample_image_metadata,
    add_cell_structure_scores,
    remove_missing_images,
    prepend_localpath,
    DEFAULT_CELLPROFILER_CSVS,
)


def run(
    cp_csv_dir="",
    csv_prefix="napari_",
    out_csv="./merged_features.csv",
    normalized_image_manifest="",
    norm_image_key="rescaled_2D_fov_tiff_path",
    fov_metadata="",
    structure_scores="",
    prepend_local=None,
    relative_columns=None,
):
    """
    Merge cellprofiler output csvs with image metadata and structure scores
    Args:
        cp_csv_dir (str): location of cellprofiler output csvs
        csv_prefix (str): optional prefix part of cellprofiler output csv file names
        out_csv (str): path to file where to save merged csv
        normalized_image_manifest (str): location of contrast stretched images that were cellprofiler input
        norm_image_key (str): column name for column in normalized image csv that contains paths to images analyzed by cellprofiler
        fov_metadata (str): location of csv file with sample metadata from labkey
        structure_scores (str): location of csv file with manual structure scores per cell
        prepend_local (NoneType or str): optional local path to prepend to relative image paths in normalized_image_manifest, so they match absolute paths in cellprofiler output
        relative_columns (NoneType or str: optional list of comma separated column names in normalized_image_manifest with relative image paths that need to be converted to absolute
    """

    image_csv = os.path.join(
        cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["image"]
    )

    # make list of ImageNumbers that failed to process in cellprofiler
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
        flag_border_cell_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["flag_border_cell"]
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
    # if normalized_image_manifest has relative image paths, convert to absolute b/f merging
    if prepend_local:
        normalized_image_manifest_abs = prepend_localpath(
            normalized_image_manifest,
            column_list=relative_columns,
            localpath=prepend_local,
        )
        new_normalized_image_manifest = "./absolute_" + os.path.basename(
            normalized_image_manifest
        )
        normalized_image_manifest_abs.to_csv(new_normalized_image_manifest, index=False)

        cp_feature_metadata_df = add_sample_image_metadata(
            cell_feature_df=cp_feature_df,
            norm_image_manifest=new_normalized_image_manifest,
            fov_metadata=fov_metadata,
        )

    # if normalized_image_manifest already has absolute image paths, able to merge using original file
    else:
        cp_feature_metadata_df = add_sample_image_metadata(
            cell_feature_df=cp_feature_df,
            norm_image_manifest=normalized_image_manifest,
            fov_metadata=fov_metadata,
        )

    # add manual structure scores to feature data frame
    feature_df = add_cell_structure_scores(
        cell_feature_df=cp_feature_metadata_df, structure_scores_csv=structure_scores
    )

    final_feature_df = remove_missing_images(feature_df)

    final_feature_df.to_csv(out_csv, index=False)


def main():
    fire.Fire(run)
