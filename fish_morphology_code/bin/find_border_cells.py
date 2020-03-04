#!/usr/bin/env python

import pandas as pd
from aicsimageio import AICSImage
import os
import fire


from fish_morphology_code.processing.merge_features.cp_processing_utils import (
    find_border_cells,
    prepend_localpath,
)


def flag_border_cells(
    normalized_image_manifest="",
    norm_image_key="rescaled_2D_fov_tiff_path",
    prepend_local=None,
    relative_columns=None,
    out_csv="./border_cells.csv",
    structure_data=True,
):
    """
    Flag napari annotated cells that touch the border of the image
    Args:
        normalized_image_manifest (str): location of contrast stretched images that were cellprofiler input
        norm_image_key (str): column name for column in normalized image csv that contains paths to images analyzed by cellprofiler
        prepend_local (NoneType or str): optional local path to prepend to relative image paths in normalized_image_manifest, so they match absolute paths in cellprofiler output
        relative_columns (NoneType or str: optional list of comma separated column names in normalized_image_manifest with relative image paths that need to be converted to absolute
        out_csv (str): path to file where to save merged csv
        structure_data (bool): True indicates that this is structure data set with shape (1, 1, 10, 1, 1736, 1776); False indicates non-structure data set with shape (1, 1, 1, 10, 624, 924)
    """

    # if normalized_image_manifest has relative image paths, convert to absolute
    if prepend_local:
        normalized_image_df = prepend_localpath(
            normalized_image_manifest,
            column_list=relative_columns,
            localpath=prepend_local,
        )
        new_normalized_image_manifest = "./absolute_" + os.path.basename(
            normalized_image_manifest
        )
        normalized_image_df.to_csv(new_normalized_image_manifest, index=False)

    else:
        normalized_image_df = pd.read_csv(normalized_image_manifest)

    # rename columns in image manifest to match cell feature df
    normalized_image_df = normalized_image_df.rename(
        columns={
            "fov_id": "FOVId",
            "original_fov_location": "fov_path",
            norm_image_key: "ImagePath",
        }
    )

    # go through fovs and make dataframe with border cells only
    images = normalized_image_df.loc[
        :, ("FOVId", "fov_path", "ImagePath")
    ].drop_duplicates()
    border_df = pd.DataFrame(
        columns=[
            "FOVId",
            "fov_path",
            "ImagePath",
            "napariCell_ObjectNumber",
            "cell_border",
        ]
    )

    for i, row in images.iterrows():
        fov_id = row["FOVId"]
        fov_path = row["fov_path"]
        image = row["ImagePath"]

        napari_annotation = AICSImage(image)

        # structure and non-structure data have different shapes
        napari_annotation_image = None
        if structure_data:
            napari_annotation_image = napari_annotation.data[0, 0, 9, 0]
        else:
            napari_annotation_image = napari_annotation.data[0, 0, 0, 9]

        border_adj_cells = find_border_cells(napari_annotation_image, 2)
        # print(border_adj_cells)
        border_only_df = pd.DataFrame(
            {
                "FOVId": fov_id,
                "fov_path": fov_path,
                "ImagePath": image,
                "napariCell_ObjectNumber": border_adj_cells,
                "cell_border": True,
            }
        )
        border_df = pd.concat([border_df, border_only_df])

    border_df = border_df.reset_index(drop=True)

    border_df.to_csv(out_csv, index=False)


def main_flag_border_cells():
    fire.Fire(flag_border_cells)
