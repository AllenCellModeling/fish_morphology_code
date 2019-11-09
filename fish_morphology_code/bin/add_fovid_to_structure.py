#!/usr/bin/env python

import fire
import pandas as pd


def run(structure_scores, normalized_image_manifest, fov_metadata, out_csv):
    r"""
        Write strucutre score csv that include image fovID
        Args:
            structure_scores (str): location (absolute path) to manual structure scores
            normalized_image_manifest (str): location (absolute path) to image manifest (ex. metadata.csv from quilt)
            fov_metadata (str): location (absolute path) of csv with fov sample metdata
            structure_scores_updated (str): location (absolute path) where to save structure csv with fov id
    """

    structure_score_df = pd.read_csv(structure_scores, index_col=0)

    # make column with image file base name in structure data frame and normalized image data frame
    structure_score_df["file_base"] = (
        structure_score_df["file_name"].str.split(".").str[0]
    )

    norm_image_df = pd.read_csv(normalized_image_manifest)

    norm_image_df["file_base"] = (
        norm_image_df["rescaled_2D_fov_tiff_path"].str.split("/").str[-1]
    )
    norm_image_df["file_base"] = (
        norm_image_df["file_base"].str.split("_").str[2:10]
    ).str.join("_")

    # merge structure and normalized image data frames by image file base name and cell id
    norm_image_structure_df = pd.merge(
        norm_image_df, structure_score_df, on=["file_base"], how="outer"
    )

    # keep only structure columns plus fov id and write structures to file
    final_structure_df = norm_image_structure_df.loc[
        :, ["fov_id", "cell_num", "file_name", "mh score", "kg score"]
    ]

    final_structure_df.to_csv(out_csv)


def main():
    fire.Fire(run)
