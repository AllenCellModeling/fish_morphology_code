#!/usr/bin/env python


import os
import fire
import pandas as pd

from fish_morphology_code.processing.merge_features.cp_processing_utils import (
    image_processing_errors,
    merge_cellprofiler_output,
    add_sample_image_metadata,
    add_cell_structure_scores,
    remove_missing_images,
    prepend_localpath,
    add_cell_border_filter,
    image_object_counts,
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
    flag_border_cell_csv=None,
    output_object_counts_csv=None,
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
        flag_border_cell_csv (NoneType or str): optional path to csv with napari cells that DO touch the border of the image
        output_object_counts_csv (NoneType or str): optionally output csv with cellprofiler object counts (from Image.csv); if provided, str must be absolute path to where csv will be saved
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
        napari_cell_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["napari_cell"]
        ),
        premerge_nuclei_centroids_csv=os.path.join(
            cp_csv_dir,
            csv_prefix + DEFAULT_CELLPROFILER_CSVS["premerge_nuclei_centroids"],
        ),
        failed_images=failed_images,
        flag_border_cell_csv=flag_border_cell_csv,
    )

    # add border cell filter here instead of in merge_cellprofiler_ouptut
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

    # optional merge: merge with napari cell border filter to flag cells that touch border of image

    if flag_border_cell_csv:
        feature_df = add_cell_border_filter(flag_border_cell_csv, feature_df)

        # move cell_border column next to finalnuc_border column
        move_cols = ["cell_border"]

        first_cols = feature_df.columns.tolist()[0:18]

        remaining_cols = [
            c for c in feature_df.columns.tolist() if c not in move_cols + first_cols
        ]

        feature_df = feature_df.loc[:, first_cols + move_cols + remaining_cols]

    final_feature_df = remove_missing_images(feature_df)

    final_feature_df.to_csv(out_csv, index=False)

    # optionally output csv with cellprofiler objects counts per image
    if output_object_counts_csv:
        object_count_df = image_object_counts(image_csv)
        object_count_df = pd.merge(
            object_count_df,
            feature_df.loc[
                :,
                (
                    "ImageNumber",
                    "FOVId",
                    "ImagePath",
                    "fov_path",
                    "probe488",
                    "probe546",
                    "probe647",
                    "plate_name",
                    "cell_line",
                    "cell_age",
                ),
            ].drop_duplicates(),
            on=["ImageNumber", "ImagePath"],
            how="outer",
        )
        first_columns = [
            "ImageNumber",
            "ImagePath",
            "FOVId",
            "fov_path",
            "probe488",
            "probe546",
            "probe647",
            "plate_name",
            "cell_line",
            "cell_age",
        ]
        count_columns = [
            c for c in object_count_df.columns.tolist() if c not in first_columns
        ]
        object_count_df = object_count_df.loc[:, (first_columns + count_columns)]

        object_count_df.to_csv(output_object_counts_csv, index=False)


def run_nonstructure(
    cp_csv_dir="",
    csv_prefix="napari_",
    out_csv="./merged_features.csv",
    normalized_image_manifest="",
    norm_image_key="rescaled_2D_fov_tiff_path",
    fov_metadata="",
    prepend_local=None,
    relative_columns=None,
    flag_border_cell_csv=None,
    output_object_counts_csv=None,
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
        flag_border_cell_csv (NoneType or str): optional path to csv with napari cells that DO touch the border of the image
        output_object_counts_csv (NoneType or str): optionally output csv with cellprofiler object counts (from Image.csv); if provided, str must be absolute path to where csv will be saved
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
        napari_cell_csv=os.path.join(
            cp_csv_dir, csv_prefix + DEFAULT_CELLPROFILER_CSVS["napari_cell"]
        ),
        premerge_nuclei_centroids_csv=os.path.join(
            cp_csv_dir,
            csv_prefix + DEFAULT_CELLPROFILER_CSVS["premerge_nuclei_centroids"],
        ),
        failed_images=failed_images,
        flag_border_cell_csv=flag_border_cell_csv,
    )

    # add border cell filter here instead of in merge_cellprofiler_ouptut
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

    # optional merge: merge with napari cell border filter to flag cells that touch border of image

    if flag_border_cell_csv:
        feature_df = add_cell_border_filter(
            flag_border_cell_csv, cp_feature_metadata_df
        )

        # move cell_border column next to finalnuc_border column
        move_cols = ["cell_border"]

        first_cols = feature_df.columns.tolist()[0:18]

        remaining_cols = [
            c for c in feature_df.columns.tolist() if c not in move_cols + first_cols
        ]

        feature_df = feature_df.loc[:, first_cols + move_cols + remaining_cols]

    final_feature_df = remove_missing_images(feature_df)

    final_feature_df.to_csv(out_csv, index=False)

    # optionally output csv with cellprofiler objects counts per image
    if output_object_counts_csv:
        object_count_df = image_object_counts(image_csv)
        object_count_df = pd.merge(
            object_count_df,
            final_feature_df.loc[
                :,
                (
                    "ImageNumber",
                    "FOVId",
                    "ImagePath",
                    "fov_path",
                    "probe488",
                    "probe546",
                    "probe647",
                    "plate_name",
                    "cell_line",
                    "cell_age",
                ),
            ].drop_duplicates(),
            on=["ImageNumber", "ImagePath"],
            how="outer",
        )
        first_columns = [
            "ImageNumber",
            "ImagePath",
            "FOVId",
            "fov_path",
            "probe488",
            "probe546",
            "probe647",
            "plate_name",
            "cell_line",
            "cell_age",
        ]
        count_columns = [
            c for c in object_count_df.columns.tolist() if c not in first_columns
        ]
        object_count_df = object_count_df.loc[:, (first_columns + count_columns)]

        object_count_df.to_csv(output_object_counts_csv, index=False)


def main():
    fire.Fire(run)
