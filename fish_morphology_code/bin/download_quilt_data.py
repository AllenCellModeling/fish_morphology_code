import warnings
import quilt3
import fire


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


def download_2D_contrasted(test=False):
    """download maxporject/seg data. if test=True, only download two randomly dampled images."""
    if test:
        download_quilt_data(
            package="rorydm/2d_autocontrasted_fields_and_single_cells_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_contrasted_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="rorydm/2d_autocontrasted_fields_and_single_cells",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_contrasted",
            ignore_warnings=True,
        )


def main_segs():
    fire.Fire(download_2D_segs)


def main_contrasted():
    fire.Fire(download_2D_contrasted)
