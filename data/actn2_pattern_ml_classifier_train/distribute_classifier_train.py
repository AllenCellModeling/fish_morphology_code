#!/usr/bin/env python


from pathlib import Path
import quilt3
import fire


def distribute_actn2_pattern_classifier_train(
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
        "matheus/assay_dev_classifier_train", "s3://allencell-internal-quilt"
    )
    internal_package.fetch(
        "/allen/aics/gene-editing/FISH/2019/assay_dev_classifier_train/"
    )
    data_dir = Path("/allen/aics/gene-editing/FISH/2019/assay_dev_classifier_train/")

    # copy contents
    for path in data_dir.iterdir():
        if path.is_dir():
            subdir = path.name
            for f in path.iterdir():
                f_name = f.name
                p.set(f"actn2_pattern_ml_classifier_train/{subdir}/{f_name}", f)
        else:
            f_name = path.name
            p.set(f"actn2_pattern_ml_classifier_train/{f_name}", path)

    # print(p["actn2_pattern_ml_classifier_train"])

    p.push(pkg_dest, s3_bucket, message="actn2_pattern_ml_classifier_train")


if __name__ == "__main__":
    fire.Fire(distribute_actn2_pattern_classifier_train)
