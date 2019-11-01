#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
import fire
import quilt3
from anndata import read_h5ad


def distribute_scrnaseq(
    test=False,
    adata_loc="/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191016.h5ad",
    dataset_name="scrnaseq_data",
    package_owner="rorydm",
    s3_bucket="s3://allencell-internal-quilt",
):

    p = quilt3.Package()

    if test:
        adata = read_h5ad(adata_loc)
        adata = adata[np.random.choice(len(adata), size=10), :]
        adata_loc_test = "tmp_scrnaseq.h5ad"
        adata.write_h5ad(adata_loc_test)
        p = p.set(Path(adata_loc).name, adata_loc_test)
        p.push(f"{package_owner}/{dataset_name}_test", s3_bucket)
        os.remove(adata_loc_test)
    else:
        p = p.set(Path(adata_loc).name, adata_loc)
        p.push(f"{package_owner}/{dataset_name}", s3_bucket)


if __name__ == "__main__":
    fire.Fire(distribute_scrnaseq)
