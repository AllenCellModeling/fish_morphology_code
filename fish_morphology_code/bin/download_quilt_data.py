import warnings
import quilt3
import fire
from random import sample, seed


def download_quilt_data(
    package="2d_segmented_fields",
    bucket="s3://allencell",
    data_save_loc="./quilt_data",
    ignore_warnings=True,
):
    """download a quilt dataset and supress nfs file attribe warnings by default"""
    quilt3.Package.install(
        "aics/integrated_transcriptomics_structural_organization_hipsc_cm",
        path=package,
        registry=bucket,
        dest=data_save_loc + "/" + package,
    )
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)

def download_2D_segs():
    """download maxproject/seg data."""
    download_quilt_data(
        package="2d_segmented_fields_fish_1",
        bucket="s3://allencell",
        data_save_loc="./quilt_data",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_segmented_fields_fish_2",
        bucket="s3://allencell",
        data_save_loc="./quilt_data",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_segmented_fields_fish_3",
        bucket="s3://allencell",
        data_save_loc="./quilt_data",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_segmented_fields_fish_4",
        bucket="s3://allencell",
        data_save_loc="./quilt_data",
        ignore_warnings=True,
    )


def download_2D_contrasted():
    """download maxproject/seg data."""
    download_quilt_data(
        package="2d_autocontrasted_fields_and_single_cells_fish_1",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_contrasted",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_fields_and_single_cells_fish_2",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_contrasted",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_fields_and_single_cells_fish_3",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_contrasted",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_fields_and_single_cells_fish_4",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_contrasted",
        ignore_warnings=True,
    )


def download_2D_features():
    """download maxproject/seg data."""
    download_quilt_data(
        package="2d_autocontrasted_single_cell_features_fish_1",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_features",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_single_cell_features_fish_2",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_features",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_single_cell_features_fish_3",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_features",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="2d_autocontrasted_single_cell_features_fish_4",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_features",
        ignore_warnings=True,
    )


def download_2D_nuclear_masks():
    """download 2D nuclear mask images."""
    download_quilt_data(
        package="2d_nuclear_masks",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_2d_nuclear_masks",
        ignore_warnings=True,
    )


def download_scrnaseq():
    """download scrnaseq data."""
    download_quilt_data(
        package="scrnaseq_data",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_scrnaseq",
        ignore_warnings=True,
    )


def download_scrnaseq_raw():
    """download scrnaseq raw fastq"""
    download_quilt_data(
        package="scrnaseq_data_raw",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_scrnaseq_raw",
        ignore_warnings=True,
    )


def download_ML_struct_scores():
    """download automated structure channel scoring."""
    download_quilt_data(
        package="automated_local_and_global_structure_fish_1",
        bucket="s3://allencell",
        data_save_loc="./fish_analysis",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="automated_local_and_global_structure_fish_2",
        bucket="s3://allencell",
        data_save_loc="./fish_analysis",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="automated_local_and_global_structure_fish_3",
        bucket="s3://allencell",
        data_save_loc="./fish_analysis",
        ignore_warnings=True,
    )
    download_quilt_data(
        package="automated_local_and_global_structure_fish_4",
        bucket="s3://allencell",
        data_save_loc="./fish_analysis",
        ignore_warnings=True,
    )


def download_nonstructure_2D_segs():
    """download non-structure maxporject/seg data."""
    download_quilt_data(
        package="2d_nonstructure_fields",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_nonstructure",
        ignore_warnings=True,
    )


def download_2D_nonstructure_features():
    """download non-structure features."""
    download_quilt_data(
        package="2d_nonstructure_single_cell_features",
        bucket="s3://allencell",
        data_save_loc="./quilt_nonstructure_features",
        ignore_warnings=True,
    )


def download_actn2_3d_seg():
    """download actn2 structure segmentation."""
    download_quilt_data(
        package="3d_actn2_segmentation",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_actn2_3d_seg",
        ignore_warnings=True,
        )


def download_probe_struc_classifier_features():
    """download features relating probes and structure classifier."""
    download_quilt_data(
        package="probe_structure_classifier",
        bucket="s3://allencell",
        data_save_loc="./quilt_probe_struc_classifier_features",
        ignore_warnings=True,
    )


def download_probe_localization_features():
    """download probe localization features."""
    download_quilt_data(
        package="probe_localization",
        bucket="s3://allencell",
        data_save_loc="./quilt_data_probe_localization_features",
        ignore_warnings=True,
    )


def main_segs():
    fire.Fire(download_2D_segs)


def main_contrasted():
    fire.Fire(download_2D_contrasted)


def main_features():
    fire.Fire(download_2D_features)


def main_scrnaseq():
    fire.Fire(download_scrnaseq)


def main_scrnaseq_raw():
    fire.Fire(download_scrnaseq_raw)


def main_MLstruct():
    fire.Fire(download_ML_struct_scores)


def main_nonstructure_segs():
    fire.Fire(download_nonstructure_2D_segs)


def main_nonstructure_features():
    fire.Fire(download_2D_nonstructure_features)


def main_probe_loc_feat():
    fire.fire(download_probe_localization_features)
