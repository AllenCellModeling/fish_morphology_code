from pathlib import Path
import subprocess

import fire
import quilt3
from anndata import read_h5ad

from fish_morphology_code.analysis.collate_plot_dataset import collate_plot_dataset


def make_small_rna_df():
    adata_rnaseq = read_h5ad(
        "../../../fish_morphology_code/quilt_data_scrnaseq/scrnaseq_cardio_20191016.h5ad"
    )

    p = quilt3.Package.browse(
        "aics/integrated_transcriptomics_structural_organization_hipsc_cm",
        registry="s3://allencell",
    )
    dest = Path("tmp_quilt_data")
    dest.mkdir(parents=True, exist_ok=True)

    fname = "29f2b5c1_scrnaseq_cardio_20191016.h5ad"
    p["scrnaseq_data/anndata/29f2b5c1_scrnaseq_cardio_20191016.h5ad"].fetch(
        dest=dest / fname
    )

    adata_rnaseq = read_h5ad(dest / fname)
    adata_rnaseq = adata_rnaseq[
        :,
        [
            "HPRT1_HUMAN",
            "COL2A1_HUMAN",
            "MYH6_HUMAN",
            "MYH7_HUMAN",
            "ATP2A2_HUMAN",
            "TCAP_HUMAN",
            "BAG3_HUMAN",
            "H19_HUMAN",
        ],
    ].copy()
    adata_rnaseq = adata_rnaseq[adata_rnaseq.obs.day.isin(["D12", "D24"]), :]
    adata_rnaseq.obs = adata_rnaseq.obs[["day", "protocol"]]

    df_rnaseq = adata_rnaseq.obs.reset_index().merge(adata_rnaseq.to_df().reset_index())
    df_rnaseq = df_rnaseq.rename(
        columns={
            "day": "Cell age",
            "protocol": "Protocol",
            "HPRT1_HUMAN": "HPRT1 expression",
            "COL2A1_HUMAN": "COL2A1 expression",
            "MYH6_HUMAN": "MYH6 expression",
            "MYH7_HUMAN": "MYH7 expression",
            "ATP2A2_HUMAN": "ATP2A2 expression",
            "TCAP_HUMAN": "TCAP expression",
            "BAG3_HUMAN": "BAG3 expression",
            "H19_HUMAN": "H19 expression",
        }
    )
    df_rnaseq["Cell age"] = df_rnaseq["Cell age"].map({"D12": "12", "D24": "24"})

    return df_rnaseq


def manuscript_plots_dataset(
    test=False,
    col_name_map={},
    dataset_name="manuscript_plots",
    package_owner="rorydm",
    readme_path="README.md",
    s3_bucket="s3://allencell-internal-quilt",
):

    df = collate_plot_dataset()
    df_rna = make_small_rna_df()

    # subsample df for a test dataset
    if test:
        df = df.sample(2, random_state=0)
        dataset_name = f"{dataset_name}_test"

    # create the dataset
    p = quilt3.Package()
    p = p.set("README.md", readme_path)
    p = p.set("data.csv", df)
    p = p.set("data_rnaseq.csv", df_rna)

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )

    # upload to quilt
    p.push(
        f"{package_owner}/{dataset_name}",
        s3_bucket,
        message=f"git commit hash of fish_morphology_code = {label}",
    )


if __name__ == "__main__":
    fire.Fire(manuscript_plots_dataset)
