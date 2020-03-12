#!/usr/bin/env python

import fire
import pandas as pd


def run(structure_scores, normalized_image_manifest, out_csv):
    r"""
        Write strucutre score csv that include image fovID
        Args:
            structure_scores (str): location (absolute path) to manual structure scores; score is actn2 organization score or myh6/7 localization score
            normalized_image_manifest (str): location (absolute path) to image manifest (ex. metadata.csv from quilt)
            structure_scores_updated (str): location (absolute path) where to save structure csv with fov id
    """

    structure_score_df = pd.read_csv(structure_scores, index_col=0)
    structure_score_df = structure_score_df.reset_index(drop=True)
    norm_image_df = pd.read_csv(normalized_image_manifest)

    # rename cell number column so it matches name in structure_score_df
    norm_image_df = norm_image_df.rename(columns={"cell_label_value": "cell_num"})

    # make column with image file base name in structure data frame and normalized image data frame
    structure_score_df["file_base"] = (
        structure_score_df["file_name"].str.split(".").str[0]
    )

    norm_image_df["file_base"] = (
        norm_image_df["rescaled_2D_fov_tiff_path"].str.split("/").str[-1]
    )
    norm_image_df["file_base"] = (
        norm_image_df["file_base"].str.split("_").str[2:10]
    ).str.join("_")

    # merge structure and normalized image data frames by image file base name and cell id
    norm_image_structure_df = pd.merge(
        norm_image_df, structure_score_df, on=["cell_num", "file_base"], how="outer"
    )

    # rename fov_id column to match name in labkey metadata
    norm_image_structure_df = norm_image_structure_df.rename(
        columns={"fov_id": "FOVId"}
    )

    # keep only structure columns plus fov id and write structures to file
    # use only columns that exist in data frame to accomodate both structure and localization scoring
    keep_col_options = set(
        [
            "FOVId",
            "cell_num",
            "file_name",
            "file_base",
            "mh score",
            "kg score",
            "probe_561_loc_score",
            "probe_638_loc_score",
        ]
    )

    keep_cols = list(keep_col_options & set(norm_image_structure_df.columns))

    # put columns in preferred order
    first_cols = ["FOVId", "cell_num", "file_name", "file_base"]

    remaining_cols = [c for c in keep_cols if c not in first_cols]

    final_structure_df = norm_image_structure_df.loc[:, first_cols + remaining_cols]

    final_structure_df.to_csv(out_csv, index=False)


def main():
    fire.Fire(run)
