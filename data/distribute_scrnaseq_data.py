#!/usr/bin/env python

from pathlib import Path
import fire
import quilt3


def distribute_scrnaseq(
    adata_loc="/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191016.h5ad",
    dataset_name="scrnaseq_data",
    package_owner="rorydm",
    s3_bucket="s3://allencell-internal-quilt",
):

    p = quilt3.Package()
    p = p.set(Path(adata_loc).name, adata_loc)

    p.push(f"{package_owner}/{dataset_name}", s3_bucket)


if __name__ == "__main__":
    fire.Fire(distribute_scrnaseq)
