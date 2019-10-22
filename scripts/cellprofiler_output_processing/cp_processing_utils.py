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
    Flag images with module processing errors
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


def mergeCellProfilerOutput(
    image_csv,
    merged_nuclei_csv,
    flag_border_nuclei_csv,
    napari_cell_csv,
    premerge_nuclei_centroids_csv,
):
    r"""
        Merge cellprofiler output csv's to combine combine all cell and nuclear features

    """
    image_df = pd.read_csv(image_csv)
    final_nuc_df = pd.read_csv(merged_nuclei_csv)
    nuc_centroid_df = pd.read_csv(premerge_nuclei_centroids_csv)
    napari_cell_df = pd.read_csv(napari_cell_csv)
    final_border_filter_df = pd.read_csv(flag_border_nuclei_csv)

    # add prefix to keep track of where object numbers are coming from => should get rid of this after doing sanity checks to check merge
    nuc_centroid_df = nuc_centroid_df.add_prefix("nuccentroid_")
    final_nuc_df = final_nuc_df.add_prefix("finalnuc_")
    napari_cell_df = napari_cell_df.add_prefix("napariCell_")
    final_border_filter_df = final_border_filter_df.add_prefix("borderfilter_")

    # only keep necessary columns
    final_nuc_df = final_nuc_df.drop(
        ["finalnuc_Children_FinalNucBorder_Count", "finalnuc_Number_Object_Number"],
        axis=1,
    )

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

    # rename columns in nuccentroid to match columns in final_nuc, so we don't end up with duplicate columns in merged df
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

    # 1st merge: merge final_nuc_df w/ nuc_centroid_df by ImageNumber and ParentFilterNuc to get
    # assignment of final nuc to napari_cell object id, which is in nuc_centroid_df
    df_link_final_with_filtered_nuc = pd.merge(
        final_nuc_df,
        nuc_centroid_df,
        on=["ImageNumber", "finalnuc_Parent_FilterNuc"],
        how="inner",
    )

    # keep feature columns at the end and metadata up front
    move_cols = [
        "finalnuc_Parent_FilterNuc",
        "nuccentroid_ObjectNumber",
        "napariCell_ObjectNumber",
    ]

    first_cols = ["ImageNumber", "finalnuc_ObjectNumber"]

    remaining_cols = [
        c
        for c in df_link_final_with_filtered_nuc.columns.tolist()
        if c not in move_cols + first_cols
    ]

    df_link_final_with_filtered_nuc = df_link_final_with_filtered_nuc.loc[
        :, first_cols + move_cols + remaining_cols
    ]

    # 2nd merge: merge with napari cell features
    df_napari_link_final_with_filtered_nuc = pd.merge(
        df_link_final_with_filtered_nuc,
        napari_cell_df,
        on=["ImageNumber", "napariCell_ObjectNumber"],
        how="outer",
    )

    # 3rd merge: merge with final nuc border filter
    complete_cell_df = pd.merge(
        df_napari_link_final_with_filtered_nuc,
        final_border_filter_df,
        on=["ImageNumber", "finalnuc_ObjectNumber"],
        how="outer",
    )

    # 4th merge: merge image paths with complete_cell_df
    final_cell_df = pd.merge(
        image_df, complete_cell_df, on=["ImageNumber"], how="outer"
    )

    return final_cell_df


def image_object_counts(image_csv):
    r"""
    Extract object count columns from image csv
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
