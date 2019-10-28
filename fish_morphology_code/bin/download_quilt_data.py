import warnings
import quilt3
import fire

from pathlib import Path


def download_quilt_data(
    package="rorydm/2d_segmented_fields",
    bucket="s3://allencell-internal-quilt",
    data_save_loc="quilt_data",
    ignore_warnings=True,
):
    """download a quilt dataset and supress nfs file attribe warnings by default"""
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)

    print(f"data_save_loc = {Path(data_save_loc).resolve()} has")
    for x in Path(data_save_loc).iterdir():
        print(x)


def download_2D_segs(test=False):
    """download maxporject/seg data. if test=True, only download two randomly dampled images."""
    if test:
        download_quilt_data(
            package="rorydm/2d_segmented_fields_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="rorydm/2d_segmented_fields",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data",
            ignore_warnings=True,
        )


def main():
    fire.Fire(download_2D_segs)
