#!/usr/bin/env python

from quilt3distribute import Dataset
import fire
import subprocess


def distribute_scrnaseq_raw(
    csv_loc="scrnaseq_data_raw.csv",
    dataset_name="scrnaseq_raw",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):

    # create the dataset
    ds = Dataset(
        dataset=csv_loc,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="README.md",
    )

    # Rename the columns on the package level
    ds.set_column_names_map(
        {"fastq_files": "fastq", "read_assignment_files": "read_assignment"}
    )

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )

    # Distribute
    ds.distribute(
        push_uri=s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_scrnaseq_raw)
