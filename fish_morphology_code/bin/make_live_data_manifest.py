#!/usr/bin/env python

import fire
import pandas as pd
import os


def run(
    structure_scores="../../data/structure_scores/KG_MH_scoring_all.csv",
    classifier_manifest="/allen/aics/assay-dev/computational/data/cardio_pipeline_datastep/local_staging_pipeline_actn2/cellfeatures/manifest_sum_projs.csv",
    plate_metadata="../../data/all_plates.csv",
    out_csv="/allen/aics/gene-editing/FISH/2019/chaos/data/20201012_actn2_live_classifier_with_metadata/live_manifest.csv",
):
    r"""
        Write structure score csv that include image fovID from labkey
        Args:
            structure_scores (str): location (absolute path) to manual structure scores; score is actn2 organization score
            classifier_manifest (str): location (absolute path) to image manifest
            plate_metadata (str): location (absolute path) of csv with plate metadata
            out_csv (str): location (absolute path) where to save manifest
    """

    classifier_df = pd.read_csv(classifier_manifest)
    classifier_df = classifier_df.rename(columns={"CellId": "cell_id"})

    classifier_df["plate"] = [
        y.split("_")[0]
        for y in [os.path.basename(x) for x in classifier_df["RawFilePath"].tolist()]
    ]
    classifier_df["well_position"] = [
        y.split(" ")[6]
        for y in [os.path.basename(x) for x in classifier_df["RawFilePath"].tolist()]
    ]
    classifier_df["image_date"] = [
        y.split("_")[2]
        for y in [os.path.basename(x) for x in classifier_df["RawFilePath"].tolist()]
    ]
    classifier_df = classifier_df.astype({"plate": "int64"})

    manual_score_df = pd.read_csv(structure_scores)
    manual_score_df = manual_score_df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
    manual_score_df = manual_score_df.loc[manual_score_df["type"] == "live"]

    # merge classifier manifest and manual scores
    merged_classifier_manual = pd.merge(
        left=classifier_df,
        right=manual_score_df,
        on=["cell_id", "RawFilePath", "plate"],
        how="outer",
    )

    merged_classifier_manual = merged_classifier_manual.drop(columns=["type"])

    # add plate metadata
    all_plates = pd.read_csv(plate_metadata)
    all_plates = all_plates.drop(columns=["wells"])
    all_plates = all_plates.loc[all_plates["type"] == "live"]

    merged_classifier_manual = merged_classifier_manual.merge(
        all_plates, on=["plate"], how="outer"
    )
    merged_classifier_manual = merged_classifier_manual.rename(
        columns={
            "plate": "plate_name",
            "age": "Cell age",
            "RawFilePath": "FOV",
            "replate": "replate_date",
        }
    )

    merged_classifier_manual.to_csv(out_csv, index=False)


if __name__ == "__main__":
    fire.Fire(run)
