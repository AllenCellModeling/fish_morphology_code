import warnings
import quilt3
import fire


def download_quilt_data(
    package="rorydm/2d_segmented_fields",
    bucket="s3://allencell-internal-quilt",
    data_save_loc="quilt_data",
    ingore_warnings=True,
):
    """download a quilt dataset and supress nfs file attribe warnings by default"""
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ingore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch("quilt_data")
    else:
        dataset_manifest.fetch("quilt_data")


def download_2D_segs_all():
    download_quilt_data(
        package="rorydm/2d_segmented_fields",
        bucket="s3://allencell-internal-quilt",
        data_save_loc="quilt_data",
        ingore_warnings=True,
    )


def download_2D_segs_test():
    download_quilt_data(
        package="rorydm/2d_segmented_fields_test",
        bucket="s3://allencell-internal-quilt",
        data_save_loc="quilt_data_test",
        ingore_warnings=True,
    )


if __name__ == "__main__":
    fire.Fire(download_quilt_data)
