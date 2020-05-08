#!/usr/bin/env python

from quilt3distribute import Dataset
import fire
import subprocess


def distribute_scrnaseq_raw(
    csv_loc="/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/scrnaseq_quilt/scrnaseq_data_raw.csv",
    dataset_name="scrnaseq_raw",
    package_owner="tanyasg",
    s3_bucket="s3://allencell-internal-quilt",
):

    # create the dataset
    ds = Dataset(
        dataset=csv_loc,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/scrnaseq_quilt/README.md",
    )

    # Rename the columns on the package level
    ds.set_column_names_map(
        {"fastq_files": "fastq", "read_assignment_files": "read_assignment"}
    )

    # add raw count matrix, cell metadata, and Seurat object (Robj) and anndata object (h5ad) as supplementary files
    ds.set_extra_files(
        [
            "/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/raw_counts.mtx",
            "/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/genes.csv",
            "/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/cells.csv",
            "/allen/aics/gene-editing/Manuscripts/Transcriptomics_2019/scrnaseq_supplement/data_package/cell_metadata.csv",
            "/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191016.h5ad",
            "/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191016.Robj",
            "/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/2019_analysis/merged_experiment_1_2/scrnaseq_cardio_20191210.RData",
        ]
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
