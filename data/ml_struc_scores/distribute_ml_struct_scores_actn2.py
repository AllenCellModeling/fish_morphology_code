#!/usr/bin/env python

import subprocess
import pandas as pd
import fire
from quilt3distribute import Dataset
from pathlib import Path


def distribute_struct_scores_actn2(
    test=False,
    csv_loc="/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/fish_morphology_code/fish_morphology_code/processing/structure_organization/results_Fish/AssayDevFishAnalsysis-Handoff-transcript2protein.csv",
    dataset_name="struct_scores_actn2",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):

    # read in original csv
    df = pd.read_csv(csv_loc)

    # only include old actn2 fish in this package -> 5500000075 B3 imaged 20190710
    date = df["original_fov_location"].str.split("/", expand=True)
    df["date"] = date[7]
    df = df[df.date == "20190710"]
    df = df.drop(columns=["date"])

    # update result image dir (moved after processing)
    img_dir = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/fish_morphology_code/fish_morphology_code/processing/structure_organization/output_Fish/"
    new_result_path = [img_dir + Path(x).name for x in df["result_image_path"].tolist()]
    df["result_image_path"] = new_result_path

    # subsample df for eg a test dataset
    if test:
        df = df.sample(2, random_state=0)
        dataset_name = f"{dataset_name}_test"

    # create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="/allen/aics/gene-editing/FISH/2019/chaos/data/20200929_classifier_features_actn2/README.md",
    )

    # set data path cols, metadata cols, and extra files
    #     ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(["result_image_path"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_struct_scores_actn2)
