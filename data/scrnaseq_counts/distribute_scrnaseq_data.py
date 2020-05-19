#!/usr/bin/env python

from quilt3distribute import Dataset
import fire
import subprocess
import numpy as np
import pandas as pd
import scipy.io


def make_test_mtx(csv_loc="scrnaseq_data_raw.csv", out_loc="./"):
    df = pd.read_csv(csv_loc)
    mtx_file = df["counts"].to_list()[0]
    cell_file = df["counts"].to_list()[2]
    cell_metadata_file = df["counts"].to_list()[3]

    raw_counts = scipy.io.mmread(mtx_file)
    cell_bc = pd.read_csv(cell_file)
    metadata_df = pd.read_csv(cell_metadata_file)

    # randomly sample 10 cells from matrix
    random_cells = np.random.choice(raw_counts.shape[1], 10, replace=False)
    random_array_sample = raw_counts.toarray()[:, random_cells]
    random_cells_bc = cell_bc.iloc[random_cells, :]
    random_cell_metadata = metadata_df.iloc[random_cells, :]

    # write random cells to file
    scipy.io.mmwrite(out_loc + "/raw_counts_test.mtx", random_array_sample)
    random_cells_bc.to_csv(out_loc + "/cells_test.csv", index=False)
    random_cell_metadata.to_csv(out_loc + "/cell_metadata_test.csv", index=False)


def distribute_scrnaseq_data(
    test=False,
    csv_loc="scrnaseq_data_raw.csv",
    dataset_name="scrnaseq_data",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):

    df = pd.read_csv(csv_loc)

    # subsample features to make test
    if test:
        # write test matrix
        make_test_mtx(csv_loc=csv_loc)

        # make test manifest; counts only; no anndata
        df = pd.DataFrame(
            {
                "counts": [
                    "raw_counts_test.mtx",
                    df["counts"][1],
                    "cells_test.csv",
                    "cells_test.csv",
                ]
            }
        )

        dataset_name = f"{dataset_name}_test"

        # create the dataset without supplementary files
        ds = Dataset(
            dataset=df,
            name=dataset_name,
            package_owner=package_owner,
            readme_path="README.md",
        )

        # columns with files to upload
        ds.set_path_columns(["counts"])

    else:
        ds = Dataset(
            dataset=df,
            name=dataset_name,
            package_owner=package_owner,
            readme_path="README.md",
        )

        # columns with files to upload
        ds.set_path_columns(["counts", "anndata"])

        # anndata object (h5ad) as supplementary files
        ds.set_extra_files(
            [
                "/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191210.RData"
            ]
        )

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_scrnaseq_data)
