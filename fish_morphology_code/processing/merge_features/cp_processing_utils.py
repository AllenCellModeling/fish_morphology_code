#!/usr/bin/env python

r"""
Process cellprofiler output to merge csvs, flag failed images, and attach image metadata
"""


import pandas as pd
import numpy as np


DEFAULT_CELLPROFILER_CSVS = {
    "image": "Image.csv",
    "merged_nuclei": "FinalNuc.csv",
    "flag_border_nuclei": "FinalNucBorder.csv",
    "napari_cell": "napari_cell.csv",
    "premerge_nuclei_centroids": "NucCentroid.csv",
}


def image_processing_errors(image_csv):
    r"""
    Identify images with cellprofiler module processing errors
    Args:
        image_csv (str): location of image csv file output by cellprofiler
    Returns:
        failed_image_numbers (numpy.ndarray): ids for failed images
    """

    image_df = pd.read_csv(image_csv)

    moduleError_cols = [
        col for col in image_df.columns.tolist() if "ModuleError" in col
    ]
    moduleError_df = image_df.loc[:, (moduleError_cols)]
    moduleStatus = [
        moduleError_df[c].values == 1 | np.isnan(moduleError_df[c].values)
        for c in moduleError_df
    ]
    image_status = np.any(moduleStatus, axis=0)

    failed_images = image_df.loc[image_status, :]
    failed_image_numbers = failed_images.ImageNumber.values

    return failed_image_numbers


def merge_cellprofiler_output(
    image_csv,
    merged_nuclei_csv,
    flag_border_nuclei_csv,
    napari_cell_csv,
    premerge_nuclei_centroids_csv,
    failed_images=None,
):
    r"""
        Merge cellprofiler output csv's to combine combine all cell and nuclear features
        Args:
            image_csv (str): location of image csv file output by cellprofiler
            merged_nuclei_csv (str): location of csv with final merged nuclei objects output by cellprofiler
            flag_border_nuclei_csv (str): location of csv with final merged nuclei objects that do not touch image border output by cellprofiler
            napari_cell_csv (str): location of csv with napari cell objects output by cellprofiler
            premerge_nuclei_centroids_csv (str): location of csv with nuclei centroid objects (b/f they get merged by napari cell) output by cellprofiler
            failed_images (NoneType or np.ndarray): optional array of failed images to flag
        Returns:
            final_feature_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler; each row is one napari cell
    """

    image_df = pd.read_csv(image_csv)
    final_nuc_df = pd.read_csv(merged_nuclei_csv)
    nuc_centroid_df = pd.read_csv(premerge_nuclei_centroids_csv)
    napari_cell_df = pd.read_csv(napari_cell_csv)
    final_border_filter_df = pd.read_csv(flag_border_nuclei_csv)

    # add prefix to keep track of where columns are coming from; will rename some later
    nuc_centroid_df = nuc_centroid_df.add_prefix("nuccentroid_")
    final_nuc_df = final_nuc_df.add_prefix("finalnuc_")
    napari_cell_df = napari_cell_df.add_prefix("napariCell_")
    final_border_filter_df = final_border_filter_df.add_prefix("borderfilter_")

    # filter unnecessary columns
    nuc_centroid_df = nuc_centroid_df.loc[
        :,
        (
            "nuccentroid_ImageNumber",
            "nuccentroid_ObjectNumber",
            "nuccentroid_Parent_FilterNuc",
            "nuccentroid_Parent_napari_cell",
        ),
    ]
    final_border_filter_df = final_border_filter_df.loc[
        :,
        (
            "borderfilter_ImageNumber",
            "borderfilter_ObjectNumber",
            "borderfilter_Parent_FinalNuc",
        ),
    ]
    image_df["ImagePath"] = (
        image_df["ObjectsPathName_napari_cell"]
        + "/"
        + image_df["ObjectsFileName_napari_cell"]
    )
    image_df = image_df.loc[:, ("ImageNumber", "ImagePath")]

    # renaming columns that will be used as keys for merging to prevent duplicates when merging
    final_nuc_df = final_nuc_df.rename(columns={"finalnuc_ImageNumber": "ImageNumber"})

    nuc_centroid_df = nuc_centroid_df.rename(
        columns={
            "nuccentroid_ImageNumber": "ImageNumber",
            "nuccentroid_Parent_FilterNuc": "finalnuc_Parent_FilterNuc",
            "nuccentroid_Parent_napari_cell": "napariCell_ObjectNumber",
        }
    )

    napari_cell_df = napari_cell_df.rename(
        columns={"napariCell_ImageNumber": "ImageNumber"}
    )

    final_border_filter_df = final_border_filter_df.rename(
        columns={
            "borderfilter_ImageNumber": "ImageNumber",
            "borderfilter_Parent_FinalNuc": "finalnuc_ObjectNumber",
        }
    )

    # 1st merge: final_nuc_df w/ nuc_centroid_df to link final nuc object id to napari_cell object id
    # inner merge b/c in cellprofiler each FinalNuc object is assigned only one Parent_FilterNuc regardless
    # of whether it's actually made up of mutliple merged nuclei; have lots of Nan's if using outer merge
    df_link_final_with_filtered_nuc = pd.merge(
        final_nuc_df,
        nuc_centroid_df,
        on=["ImageNumber", "finalnuc_Parent_FilterNuc"],
        how="inner",
    )

    # 2nd merge: merge with napari cell features
    df_napari_link_final_with_filtered_nuc = pd.merge(
        df_link_final_with_filtered_nuc,
        napari_cell_df,
        on=["ImageNumber", "napariCell_ObjectNumber"],
        how="outer",
    )

    # 3rd merge: merge with final nuc border filter to flag FinalNuc that touch border of image
    cell_feature_df = pd.merge(
        df_napari_link_final_with_filtered_nuc,
        final_border_filter_df,
        on=["ImageNumber", "finalnuc_ObjectNumber"],
        how="outer",
    )
    # rename border filter column and convert to boolean
    cell_feature_df = cell_feature_df.rename(
        columns={"borderfilter_ObjectNumber": "finalnuc_border"}
    )
    cell_feature_df["finalnuc_border"] = cell_feature_df["finalnuc_border"].isna()

    # 4th merge: merge input image paths with features data frame
    cell_feature_image_df = pd.merge(
        image_df, cell_feature_df, on=["ImageNumber"], how="outer"
    )

    # Clean up and re-order columns

    # rename nuclei count column
    cell_feature_image_df = cell_feature_image_df.rename(
        columns={"napariCell_Children_NucCentroid_Count": "napariCell_nuclei_Count"}
    )

    # remove unnecessary columns
    cell_feature_image_df = cell_feature_image_df.drop(
        [
            "finalnuc_Children_FinalNucBorder_Count",
            "finalnuc_Number_Object_Number",
            "finalnuc_Mean_seg_probe_561_Number_Object_Number",
            "finalnuc_Mean_seg_probe_638_Number_Object_Number",
            "napariCell_Mean_seg_probe_561_Number_Object_Number",
            "napariCell_Mean_seg_probe_638_Number_Object_Number",
            "napariCell_Number_Object_Number",
        ],
        axis=1,
    )
    # keep feature columns at the end and metadata up front
    move_cols = [
        "napariCell_nuclei_Count",
        "finalnuc_border",
        "finalnuc_ObjectNumber",
        "finalnuc_Parent_FilterNuc",
        "nuccentroid_ObjectNumber",
    ]

    # optional flagging of failed images
    if failed_images is None:
        cell_feature_image_df["ImageFailed"] = np.NaN
    else:
        cell_feature_image_df["ImageFailed"] = np.isin(
            cell_feature_image_df["ImageNumber"].values, failed_images
        )

    first_cols = ["ImageNumber", "ImagePath", "ImageFailed", "napariCell_ObjectNumber"]

    remaining_cols = [
        c
        for c in cell_feature_image_df.columns.tolist()
        if c not in move_cols + first_cols
    ]

    final_feature_df = cell_feature_image_df.loc[
        :, first_cols + move_cols + remaining_cols
    ]

    return final_feature_df


def image_object_counts(image_csv):
    r"""
    Extract object counts per image from cellprofiler image csv
    Args:
        image_csv (str): location of image csv file output by cellprofiler
    Returns:
        image_df (pd.DataFrame): counts of objects per image (fov)
    """

    image_df = pd.read_csv(image_csv)
    image_df["ImagePath"] = (
        image_df["ObjectsPathName_napari_cell"]
        + "/"
        + image_df["ObjectsFileName_napari_cell"]
    )
    image_df = image_df.loc[
        :,
        (
            "ImageNumber",
            "ImagePath",
            "Count_FilterNuc",
            "Count_FilterNucBorder",
            "Count_FinalNuc",
            "Count_FinalNucBorder",
            "Count_MergedNucCentroidByCell",
            "Count_NucBubble",
            "Count_NucCentroid",
            "Count_napari_cell",
            "Count_nuc",
            "Count_seg_probe_561",
            "Count_seg_probe_638",
        ),
    ]
    return image_df


def add_sample_image_metadata(
    cell_feature_df, norm_image_manifest, fov_raw_seg, fov_metadata
):
    r"""
        Add fish sample and image metadata to cell feature data frame
        Args:
            cell_feature_df (pd.DataFrame): cellprofiler cell and nuclei features merged by merge_cellprofiler_output
            norm_image_manifest (str): location of csv with paths to fov pre-normalization and normalized tiffs that were analyzed with cellprofiler
            fov_raw_seg (str): location of csv with paths to fov raw tiffs and processed but pre-normalization tiffs
            fov_metadata (str): location of csv with fov sample metadata
        Returns:
            final_feature_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler with columns added for sample and image metadata; each row is one napari cell
    """

    sample_metadata_df = pd.read_csv(fov_metadata)
    norm_image_manifest_df = pd.read_csv(norm_image_manifest)
    raw_seg_image_df = pd.read_csv(fov_raw_seg)

    # use normalized image manifest to get un-normalized image paths and merge
    # single cell image paths into cell feature data frame

    # rename columns in image manifest to match cell feature df
    norm_image_manifest_df = norm_image_manifest_df.rename(
        columns={
            "field_image_path": "seg_file_name",
            "rescaled_field_image_path": "ImagePath",
            "cell_label_value": "napariCell_ObjectNumber",
        }
    )
    cell_feature_image_df = pd.merge(
        cell_feature_df,
        norm_image_manifest_df,
        on=["ImagePath", "napariCell_ObjectNumber"],
        how="outer",
    )

    # get raw image fov path
    cell_feature_image_raw_df = pd.merge(
        cell_feature_image_df, raw_seg_image_df, on=["seg_file_name"], how="outer"
    )

    # merge sample metadata into cell_feature_df
    cell_feature_image_metadata_df = pd.merge(
        cell_feature_image_raw_df,
        sample_metadata_df,
        on=["fov_path", "FOVId"],
        how="outer",
    )
    # drop unnecessary columns
    cell_feature_image_metadata_df = cell_feature_image_metadata_df.drop(
        ["seg_file_name", "FOVId", "ge_wellID", "notes"], axis=1
    )

    move_cols = [
        "single_cell_channel_output_path",
        "fov_path",
        "well_position",
        "microscope",
        "image_date",
        "probe488",
        "probe546",
        "probe647",
        "plate_name",
        "cell_line",
        "cell_age",
    ]

    first_cols = ["ImageNumber", "ImagePath", "ImageFailed", "napariCell_ObjectNumber"]

    remaining_cols = [
        c
        for c in cell_feature_image_metadata_df.columns.tolist()
        if c not in move_cols + first_cols
    ]

    final_feature_df = cell_feature_image_metadata_df.loc[
        :, first_cols + move_cols + remaining_cols
    ]
    return final_feature_df


def add_cell_structure_scores(
    cell_feature_df, structure_score_df, norm_image_suffix="_rescaled.ome.tiff"
):
    """
        Add manual sarcomere structure scores to cell feature data frame
        Args:
            cell_feature_df (pd.DataFrame): cellprofiler cell and nuclei features merged by merge_cellprofiler_output
            structure_score_df (pd.DataFrame): manual sarcomere structure scores per napari cell; cell_num = napariCell_ObjectNumber
            norm_image_suffix (str): suffix added to normalized tiff image names
        Returns:
            cell_feature_score_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler with manual structure scores added
    """

    # rename columns to match cell feature columns
    structure_score_df = structure_score_df.rename(
        columns={
            "cell_num": "napariCell_ObjectNumber",
            "mh_score": "mh_structure_org_score",
            "kg_score": "kg_structure_org_score",
        }
    )
    structure_score_df["file_base"] = (
        structure_score_df["file_name"].str.split(".").str[0]
    )
    cell_feature_df["file_base"] = (
        cell_feature_df["ImagePath"]
        .str.split("/")
        .str[-1]
        .str[0 : -len(norm_image_suffix)]
    )

    cell_feature_score_df = pd.merge(
        cell_feature_df,
        structure_score_df,
        on=["file_base", "napariCell_ObjectNumber"],
        how="outer",
    )
    cell_feature_score_df = cell_feature_score_df.drop(
        ["file_base", "file_name"], axis=1
    )

    return cell_feature_score_df


def cat_structure_scores(score_files):
    """
        Concatenate scores when they are split by plate into multiple csvs
        Args:
            score_files (str): location of csv that contains the paths to each score csv
        Returns:
            structure_scores_df (pd.DataFrame): concatenated scores from all plates
    """

    file_df = pd.read_csv(score_files)

    score_df_list = [pd.read_csv(f) for f in file_df["score_path"].tolist()]

    structure_scores_df = pd.concat(score_df_list, axis=0, ignore_index=True)

    return structure_scores_df


def remove_missing_images(feature_df):
    r"""
        Remove rows from feature data frame that have an undefined ImageNumber (image is one of the manifests but wasn't part of cellprofiler run"
        Args:
            feature_df (pd.DataFrame): cell features including column with ImageNumber
        Returns:
            feature_df_clean (pd.DataFrame): cell features without undefined ImageNumber rows
    """

    missing_images = feature_df["ImageNumber"].isna()
    feature_df_clean = feature_df.loc[~missing_images, :]

    return feature_df_clean
