#!/usr/bin/env python

import subprocess
import pandas as pd
import fire
from quilt3distribute import Dataset


def distribute_struct_scores_actn2_live(
    test=False,
    csv_loc="/allen/aics/assay-dev/computational/data/cardio_pipeline_datastep/local_staging_pipeline_actn2/cellfeatures/manifest.csv",
    dataset_name="struct_scores_actn2_live",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):

    # read in original csv
    df = pd.read_csv(csv_loc)
    df["CellPath"] = df["CellPath"].str.replace(
        "singlecells",
        "/allen/aics/assay-dev/computational/data/cardio_pipeline_datastep/local_staging_pipeline_actn2/singlecells/singlecells",
        regex=False,
    )

    # subsample df for eg a test dataset
    if test:
        df = df.sample(2, random_state=0)
        dataset_name = f"{dataset_name}_test"

    # create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="/allen/aics/gene-editing/FISH/2019/chaos/data/20200929_classifier_features_actn2/README_actn2_live.md",
    )

    # set data path cols, metadata cols, and extra files
    #     ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(["CellPath"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_struct_scores_actn2_live)
