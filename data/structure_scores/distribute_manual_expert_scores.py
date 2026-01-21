#!/usr/bin/env python


from pathlib import Path
import quilt3
import fire


def distribute_manual_expert_scores(
    pkg_dest="aics/integrated_transcriptomics_structural_organization_hipsc_cm",
    s3_bucket="s3://allencell",
    edit=True,
):

    # either edit package if it exists or create new
    if edit:
        p = quilt3.Package.browse(pkg_dest, registry=s3_bucket)
    else:
        p = quilt3.Package()

    # fetch internal package
    internal_package = quilt3.Package.browse(
        "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
    )
    internal_package.fetch("/allen/aics/gene-editing/FISH/2019/manual_expert_scores/")
    data_dir = Path("/allen/aics/gene-editing/FISH/2019/manual_expert_scores/")

    # copy contents
    for path in data_dir.iterdir():
        if path.is_dir():
            subdir = path.name
            for f in path.iterdir():
                f_name = f.name
                p.set(f"manual_expert_scores/{subdir}/{f_name}", f)
        else:
            f_name = path.name
            p.set(f"manual_expert_scores/{f_name}", path)

    # print(p["manual_expert_scores"])

    p.push(pkg_dest, s3_bucket, message="manual_expert_scores")


if __name__ == "__main__":
    fire.Fire(distribute_manual_expert_scores)
