import quilt3
import fire
from pathlib import Path


def make_pkg_maps(
    all_pkgs=[
        "calystay/2d_nuclear_masks",
        "calystay/3d_actn2_segmentation",
        "calystay/probe_localization",
        "calystay/probe_structure_classifier",
        "calystay/segmented_nuc_labels",
        "matheus/assay_dev_fish_analysis",
        "rorydm/scrnaseq_data",
        "tanyasg/2d_autocontrasted_single_cell_features",
        "tanyasg/2d_nonstructure_fields",
        "tanyasg/2d_nonstructure_single_cell_features",
        "tanyasg/scrnaseq_raw",
        "rorydm/2d_autocontrasted_fields_and_single_cells",
        "rorydm/2d_segmented_fields",
        "rorydm/manuscript_plots",
    ],
    skip_test_pkgs=["tanyasg/scrnaseq_raw_test", "rorydm/manuscript_plots_test"],
    manual_rename={
        "matheus/assay_dev_fish_analysis": "automated_local_and_global_structure"
    },
    internal_s3_url="s3://allencell-internal-quilt",
):

    # grab a list of packages that actually exist internally
    pkg_list = list(quilt3.list_packages("s3://allencell-internal-quilt"))

    # create dictionary for original fully qualified name to simple name
    pkg_map = {p: p.split("/")[-1] for p in all_pkgs}

    # manually fix up some opaque names
    for k, v in manual_rename.items():
        pkg_map[k] = v

    # make the map for test pkgs
    pkg_map_test = {f"{k}_test": v for k, v in pkg_map.items()}

    # make sure every key has a unique value
    assert len(set(pkg_map.values())) == len(pkg_map)
    assert len(set(pkg_map_test.values())) == len(pkg_map_test)

    # make sure all our keys are in the s3 bucket, skipping some _test versions that don't exist
    assert len([k for k, v in pkg_map.items() if k in pkg_list]) == len(pkg_map)
    assert len([k for k, v in pkg_map_test.items() if k in pkg_list]) == len(
        pkg_map_test
    ) - len(["tanyasg/scrnaseq_raw_test", "rorydm/manuscript_plots_test"])

    # once checked, filter on only what's there
    pkg_map = {k: v for k, v in pkg_map.items() if k in pkg_list}
    pkg_map_test = {k: v for k, v in pkg_map_test.items() if k in pkg_list}

    return pkg_map, pkg_map_test


def aggregate_and_push(
    pkg_map,
    source_S3_url="s3://allencell-internal-quilt",
    dest_S3_url="s3://allencell-internal-quilt",
    dest_pkg_name="aics/cardio_diff_manuscript",
):
    # real data
    q = quilt3.Package()
    # TODO add readme to top level package

    for (low_level_pkg_str, new_subdir) in pkg_map.items():
        p = quilt3.Package.browse(low_level_pkg_str, source_S3_url)
        for (logical_key, physical_key) in p.walk():
            q.set(f"{new_subdir}/{logical_key}"), physical_key)

    q.push(dest_pkg_name, dest_S3_url, message="FISH data reorg")


PKG_MAP, PKG_MAP_TEST = make_pkg_maps()


def agg_push_test():
    aggregate_and_push(
        PKG_MAP_TEST,
        source_S3_url="s3://allencell-internal-quilt",
        dest_S3_url="s3://allencell-internal-quilt",
        dest_pkg_name="aics/cardio_diff_manuscript_test",
    )


def agg_push_full():
    aggregate_and_push(
        PKG_MAP,
        source_S3_url="s3://allencell-internal-quilt",
        dest_S3_url="s3://allencell-internal-quilt",
        dest_pkg_name="aics/cardio_diff_manuscript",
    )


def main_consolidate_full():
    fire.Fire(agg_push_full)


def main_consolidate_test():
    fire.Fire(agg_push_test)
