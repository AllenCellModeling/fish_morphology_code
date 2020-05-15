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


def download_2D_features(test=False):
    """download maxporject/seg data. if test=True, only download two randomly dampled images."""
    if test:
        download_quilt_data(
            package="tanyasg/2d_autocontrasted_single_cell_features_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_features_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="tanyasg/2d_autocontrasted_single_cell_features",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_features",
            ignore_warnings=True,
        )


def download_2D_nuclear_masks(test=False):
    """download 2D nuclear mask images. if test=True, only download two randomly sampled images"""
    if test:
        download_quilt_data(
            package="calystay/2d_nuclear_masks_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_2d_nuclear_masks_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="calystay/2d_nuclear_masks",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_2d_nuclear_masks",
            ignore_warnings=True,
        )


def download_scrnaseq(test=False):
    """download scrnaseq data. if test=True, only download counts for ten random cells."""
    if test:
        download_quilt_data(
            package="rorydm/scrnaseq_data_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_scrnaseq_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="rorydm/scrnaseq_data",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_scrnaseq",
            ignore_warnings=True,
        )


def download_scrnaseq_raw(test=False):
    """download scrnaseq raw data"""
    download_quilt_data(
        package="tanyasg/scrnaseq_data_raw",
        bucket="s3://allencell-internal-quilt",
        data_save_loc="quilt_data_scrnaseq_raw",
        ignore_warnings=True,
    )

    download_quilt_data(
        package="tanyasg/scrnaseq_data_raw",
        bucket="s3://allencell-internal-quilt",
        data_save_loc="quilt_data_scrnaseq_raw",
        ignore_warnings=True,
    )


def download_ML_struct_scores(test=False):
    """download automated structure channel scoring."""
    if test:
        download_quilt_data(
            package="matheus/assay_dev_fish_analysis_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_matheus_assay_dev_fish_analysis_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="matheus/assay_dev_fish_analysis",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_matheus_assay_dev_fish_analysis",
            ignore_warnings=True,
        )


def download_nonstructure_2D_segs(test=False):
    """download non-structure maxporject/seg data. if test=True, only download two randomly sampled images."""
    if test:
        download_quilt_data(
            package="tanyasg/2d_nonstructure_fields_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_nonstructure_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="tanyasg/2d_nonstructure_fields",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_nonstructure",
            ignore_warnings=True,
        )


def download_2D_nonstructure_features(test=False):
    """download non-structure features. if test=True, only download features for two randomly sampled images."""
    if test:
        download_quilt_data(
            package="tanyasg/2d_nonstructure_single_cell_features_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_nonstructure_features_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="tanyasg/2d_nonstructure_single_cell_features",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_nonstructure_features",
            ignore_warnings=True,
        )

  
def download_actn2_3d_seg(test=False):
    """download actn2 structure segmentation. if test=True, only download two randomly sampled images."""
    if test:
        download_quilt_data(
            package="calystay/3d_actn2_segmentation_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_actn2_3d_seg_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="calystay/3d_actn2_segmentation",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_actn2_3d_seg",
            ignore_warnings=True,
        )


def download_probe_struc_classifier_features(test=False):
    """download features relating probes and structure classifier. if test=True, only download features for two randomly sampled images."""
    if test:
        download_quilt_data(
            package="calystay/probe_structure_classifier_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_probe_struc_classifier_features_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="calystay/probe_structure_classifier",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_probe_struc_classifier_features",
            ignore_warnings=True,
        )


def download_probe_localization_features(test=False):
    """download probe localization features. if test=True, only download two randomly sampled images."""
    if test:
        download_quilt_data(
            package="calystay/probe_localization_test",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_probe_localization_features_test",
            ignore_warnings=True,
        )
    else:
        download_quilt_data(
            package="calystay/probe_localization",
            bucket="s3://allencell-internal-quilt",
            data_save_loc="quilt_data_probe_localization_features",
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
