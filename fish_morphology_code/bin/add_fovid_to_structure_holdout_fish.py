#!/usr/bin/env python

import fire
import pandas as pd


def run(
    structure_scores="/allen/aics/gene-editing/FISH/2019/chaos/image_processing_scripts/fish_morphology_code/data/structure_scores/KG_MH_scoring_all.csv",
    classifier_manifest="/allen/aics/assay-dev/computational/data/cardio_pipeline_datastep/local_staging_new_fish_max/singlecells/manifest.csv",
    zero_scores="/allen/aics/gene-editing/FISH/2019/chaos/image_processing_scripts/fish_morphology_code/data/structure_scores/scores_bonus_fish_metadata.csv",
    out_csv="/allen/aics/gene-editing/FISH/2019/chaos/image_processing_scripts/fish_morphology_code/data/structure_scores/holdout_fish_manual_scores.csv",
):
    r"""
        Write structure score csv that include image fovID from labkey
        Args:
            structure_scores (str): location (absolute path) to manual structure scores; score is actn2 organization score
            classifier_manifest (str): location (absolute path) to image manifest; image manifest must include both labkey fovID
            and cell ids used in structure classifier pipeline
            zero_scores (str): location (absolute path) to manual 0 structure scores; only 0 scores are indicated; all other are NaN
            structure_scores_updated (str): location (absolute path) where to save structure csv with fov id
    """

    manual_score_df = pd.read_csv(structure_scores)
    manual_score_df = manual_score_df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

    # grab only fish manual scores
    manual_score_df = manual_score_df.loc[manual_score_df["type"] == "fish"].copy()

    manual_score_df = manual_score_df.rename(columns={"plate": "plate_name"})
    manual_score_df = manual_score_df.astype({"plate_name": "object"})

    fish_classifier_manifest = pd.read_csv(classifier_manifest)
    fish_classifier_manifest = fish_classifier_manifest.drop(
        columns=[
            "Unnamed: 0",
            "2D_fov_tiff_path",
            "rescaled_2D_fov_tiff_path",
            "rescaled_2D_single_cell_tiff_path",
        ]
    )

    # merge on classifier pipeline assigned CellPath
    fish_classifier_manifest = fish_classifier_manifest.merge(
        manual_score_df, on="CellPath", how="inner"
    )

    fish_classifier_manifest = fish_classifier_manifest.rename(
        columns={
            "fov_id": "FOVId",
            "cell_label_value": "napariCell_ObjectNumber",
            "original_fov_location": "fov_path",
            "MH_score": "mh score",
            "KG_score": "kg score",
            "path": "classifier_image_path",
            "cell_id": "classifier_cell_id",
        }
    )
    fish_classifier_manifest = fish_classifier_manifest.drop(
        columns=[
            "fov_path",
            "plate_name",
            "RawFilePath",
            "classified_fov",
            "CellPath",
            "type",
            "image_name",
            "cell_id_filename",
        ]
    )

    # add manual zero structure score column if it exists
    if len(zero_scores) > 0:
        zero_scores_df = pd.read_csv(zero_scores)
        zero_scores_df = zero_scores_df.drop(columns=["Unnamed: 0"])
        zero_scores_df = zero_scores_df.rename(
            columns={
                "fov_id": "FOVId",
                "cell_label_value": "napariCell_ObjectNumber",
                "score": "no_structure",
            }
        )
        merged_scores = pd.merge(
            zero_scores_df,
            fish_classifier_manifest,
            on=["FOVId", "napariCell_ObjectNumber"],
            how="outer",
        )

        merged_scores.to_csv(out_csv)

    else:
        fish_classifier_manifest.to_csv(out_csv)


if __name__ == "__main__":
    fire.Fire(run)
