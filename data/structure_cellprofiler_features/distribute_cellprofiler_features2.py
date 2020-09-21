#!/usr/bin/env python

from quilt3distribute import Dataset
import fire
import subprocess
import numpy as np
import pandas as pd


def make_test_csv(
    csv_loc="/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20200903/merged_features/features2quilt/features2quilt.csv",
    out_loc="./",
):
    df = pd.read_csv(csv_loc)
    feature_csv = df["feature_file"].to_list()[0]
    object_count_csv = df["image_object_count_file"].to_list()[0]
    feature_df = pd.read_csv(feature_csv)
    object_df = pd.read_csv(object_count_csv)

    image_sample = np.random.choice(
        feature_df["ImageNumber"].unique(), size=2, replace=False
    )
    feature_df = feature_df.loc[feature_df["ImageNumber"].isin(image_sample), :]
    object_df = object_df.loc[object_df["ImageNumber"].isin(image_sample), :]
    feature_df.to_csv(out_loc + "/cp_features_test.csv", index=False)
    object_df.to_csv(out_loc + "/image_object_counts_test.csv", index=False)


def distribute_cellprofiler_features(
    test=False,
    csv_loc="/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20200903/merged_features/features2quilt/features2quilt.csv",
    dataset_name="2d_autocontrasted_single_cell_features2",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):
    df = pd.read_csv(csv_loc)

    # subsample features to make test
    if test:
        # write test feature csv and test image counts csv
        make_test_csv(csv_loc=csv_loc)
        cell_line = df["cell_line"][0]
        cellprofiler_id = df["cellprofiler_id"][0]

        # make test manifest
        df = pd.DataFrame(
            {
                "feature_file": ["cp_features_test.csv"],
                "image_object_count_file": ["image_object_counts_test.csv"],
                "cell_line": [cell_line],
                "cellprofiler_id": [cellprofiler_id],
            }
        )

        dataset_name = f"{dataset_name}_test"

    # Create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="README.md",
    )

    # Optionally add common additional requirements
    ds.add_usage_doc("https://docs.quiltdata.com/walkthrough/reading-from-a-package")
    ds.add_license("https://www.allencell.org/terms-of-use.html")

    # Optionally indicate column values to use for file metadata
    ds.set_metadata_columns(["cell_line", "cellprofiler_id"])

    # Optionally rename the columns on the package level
    ds.set_column_names_map(
        {"feature_file": "features", "image_object_count_file": "object_counts"}
    )

    # add commit hash to message
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    # Distribute
    ds.distribute(
        push_uri=s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_cellprofiler_features)
