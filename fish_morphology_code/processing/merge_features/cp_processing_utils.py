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
    flag_border_cell_csv=None,
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
            "Count_NucBubble",
            "Count_napari_cell",
            "Count_nuc",
            "Count_seg_probe_561",
            "Count_seg_probe_638",
        ),
    ]
    return image_df


def add_sample_image_metadata(
    cell_feature_df,
    norm_image_manifest,
    fov_metadata,
    norm_image_key="rescaled_2D_fov_tiff_path",
):
    r"""
        Add fish sample and image metadata to cell feature data frame
        Args:
            cell_feature_df (pd.DataFrame): cellprofiler cell and nuclei features merged by merge_cellprofiler_output
            norm_image_manifest (str): location of csv with paths to fov pre-normalization and normalized tiffs that were analyzed with cellprofiler
            fov_metadata (str): location of csv with fov sample metadata
            norm_image_key (str): column name for column in norm_image_manifest csv that contains path to images that were analyzed with cellprofiler
        Returns:
            final_feature_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler with columns added for sample and image metadata; each row is one napari cell
    """

    sample_metadata_df = pd.read_csv(fov_metadata)
    norm_image_manifest_df = pd.read_csv(norm_image_manifest)

    # use normalized image manifest to get un-normalized image paths and merge
    # single cell image paths into cell feature data frame

    # rename columns in image manifest to match cell feature df
    norm_image_manifest_df = norm_image_manifest_df.rename(
        columns={
            "fov_id": "FOVId",
            "original_fov_location": "fov_path",
            norm_image_key: "ImagePath",
            "cell_label_value": "napariCell_ObjectNumber",
        }
    )

    # merge 1: add raw fov path to cell_feature_df
    cell_feature_image_df = pd.merge(
        cell_feature_df,
        norm_image_manifest_df,
        on=["ImagePath", "napariCell_ObjectNumber"],
        how="outer",
    )

    # merge 2: use raw fov path to add sample metadata into cell_feature_df
    cell_feature_image_metadata_df = pd.merge(
        cell_feature_image_df, sample_metadata_df, on=["fov_path", "FOVId"], how="outer"
    )
    # drop unnecessary columns
    cell_feature_image_metadata_df = cell_feature_image_metadata_df.drop(
        ["ge_wellID", "notes"], axis=1
    )

    move_cols = [
        "rescaled_2D_single_cell_tiff_path",
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

    first_cols = [
        "ImageNumber",
        "FOVId",
        "ImagePath",
        "ImageFailed",
        "napariCell_ObjectNumber",
    ]

    remaining_cols = [
        c
        for c in cell_feature_image_metadata_df.columns.tolist()
        if c not in move_cols + first_cols
    ]

    final_feature_df = cell_feature_image_metadata_df.loc[
        :, first_cols + move_cols + remaining_cols
    ]
    return final_feature_df


def add_cell_structure_scores(cell_feature_df, structure_scores_csv):
    """
        Add manual sarcomere structure scores to cell feature data frame
        Args:
            cell_feature_df (pd.DataFrame): cellprofiler cell and nuclei features merged by merge_cellprofiler_output
            structure_scores_csv (str): location of csv file with manual sarcomere structure scores per napari cell; cell_num = napariCell_ObjectNumber
        Returns:
            cell_feature_score_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler with manual structure scores added
    """

    # rename columns to match cell feature columns
    structure_score_df = pd.read_csv(structure_scores_csv, index_col=0)
    structure_score_df = structure_score_df.rename(
        columns={
            "cell_num": "napariCell_ObjectNumber",
            "mh score": "mh_structure_org_score",
            "kg score": "kg_structure_org_score",
        }
    )

    cell_feature_score_df = pd.merge(
        cell_feature_df,
        structure_score_df,
        on=["FOVId", "napariCell_ObjectNumber"],
        how="outer",
    )
    cell_feature_score_df = cell_feature_score_df.drop(
        ["file_name", "file_base"], axis=1
    )

    return cell_feature_score_df


def add_cell_probe_localization_scores(cell_feature_df, probe_localization_scores_csv):
    """
        Add manual probe localization scores to cell feature data frame
        Args:
            cell_feature_df (pd.DataFrame): cellprofiler cell and nuclei features merged by merge_cellprofiler_output
            probe_localization_scores_csv (str): location of csv file with manual probe localization scores per napari cell; cell_num = napariCell_ObjectNumber
        Returns:
            cell_feature_score_df (pd.DataFrame): all cell and nuclei features calculated by cellprofiler with manual probe localization scores added
    """

    # rename columns to match cell feature columns
    localization_score_df = pd.read_csv(probe_localization_scores_csv, index_col=0)
    localization_score_df = localization_score_df.rename(
        columns={"cell_num": "napariCell_ObjectNumber"}
    )

    cell_feature_score_df = pd.merge(
        cell_feature_df,
        localization_score_df,
        on=["FOVId", "napariCell_ObjectNumber"],
        how="outer",
    )
    cell_feature_score_df = cell_feature_score_df.drop(
        ["file_name", "file_base"], axis=1
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


def prepend_localpath(image_csv, column_list, localpath):
    r"""
        Convert relative image paths in image_csv to absolute paths
        Args:
            image_csv (str): csv file where one or more columns with relative image paths (ex. quilt metadata.csv)
            column_list (list): list of column names in csv that contain relative paths to be converted
            localpath (str): prepend this path to relative image paths to convert to absolute (ex. rescaled_2D_tiff/5500000013_rescaled.ome.tiff -> /home/tanyag/quilt_data_contrasted/rescaled_2D_tiff/5500000013_rescaled.ome.tiff)
        Returns:
            image_path_df (pd.DataFrame): image path data frame with absolute image paths
    """

    image_path_df = pd.read_csv(image_csv)

    # prepend localpath to columns specified in column_list
    for c in column_list:
        image_path_df[c] = localpath + "/" + image_path_df[c]

    return image_path_df


def find_border_cells(image_array, border_buffer_fix=1):
    r"""
        Use napari cell annotations to identify cells that touch the border of the image. Right and bottom of array have extra pixels added by Napari (?) that need to be removed to correctly find cells touching border.
        Args:
            image_array (array): numpy array with cell annotation labels
            border_buffer_fix (int): number of pixels on right and bottom of array that need to be removed to correctly flag border cells touching right and bottom of image
        Returns:
            border_cells (array): labels of cells touching the border of the image
    """

    # make new array with same dimensions as image_array and set border pixels to 1
    x = np.zeros_like(image_array)

    x[0] = 1
    x[
        -border_buffer_fix - 1 :
    ] = 1  # add buffer to bottom border to fix napari exta pixels
    x[:, 0] = 1
    x[
        :, -border_buffer_fix - 1 :
    ] = 1  # add buffer to right border to fix napari exta pixels

    # find cell labels that fall within border buffer zone
    border_labels = np.unique(image_array * x)
    border_labels = border_labels[border_labels != 0]

    return border_labels


def add_cell_border_filter(flag_border_cell_csv, cell_feature_df):
    r"""
        Add border cell filter to feature data frame by merging by FOVId and napari cell object number
        Args:
            flag_border_cell_csv (str): absolute path to csv with napari annotated cells that touch the border of the image
            feature_df (pd.DataFrame): merged features from cellprofiler output; must include FOVId, ImagePath, fov_path, and napariCell_ObjectNumber columns for merging with border filter csv
        Returns:
            features_cellborder_filter_df (pd.DataFrame): cell profiler features with column added for border cell filter

    """

    # optional merge: merge with napari cell border filter to flag cells that touch border of image
    cell_border_filter_df = pd.read_csv(flag_border_cell_csv)

    features_cellborder_filter_df = pd.merge(
        cell_feature_df,
        cell_border_filter_df,
        on=["FOVId", "ImagePath", "fov_path", "napariCell_ObjectNumber"],
        how="outer",
    )

    features_cellborder_filter_df["cell_border"] = ~features_cellborder_filter_df[
        "cell_border"
    ].isna()

    return features_cellborder_filter_df
