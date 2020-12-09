#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import boto3
import quilt3
import fire


def update_quilt_readme(
    pkg_name="calystay/probe_localization",
    s3_bucket="s3://allencell-internal-quilt",
    new_readme_loc=Path("data/probe_localization/README.md"),
):
    """Quickly update a package's README without ahving to run all the distribute code."""
    p = quilt3.Package.browse(pkg_name, s3_bucket)
    p.set("README.md", new_readme_loc)
    p.push(pkg_name, s3_bucket)


README_MAP = {
    "rorydm/2d_autocontrasted_fields_and_single_cells": "data/2d_autocontrasted_fields_and_single_cells/README.md",
    "tanyasg/2d_autocontrasted_single_cell_features": "data/structure_cellprofiler_features/README.md",
    "tanyasg/2d_nonstructure_fields": "data/nonstructure_fields/README.md",
    "tanyasg/2d_nonstructure_single_cell_features": "data/nonstructure_features/README.md",
    "calystay/2d_nuclear_masks": "data/nuclear_masks/README.md",
    "rorydm/2d_segmented_fields": "data/2d_segmented_fields/README.md",
    "calystay/3d_actn2_segmentation": "data/actn2_3d_seg/README.md",
    "matheus/assay_dev_fish_analysis": "fish_morphology_code/processing/structure_organization/tools/assay-dev-fish.md",
    "rorydm/manuscript_plots": "data/plots_dataset/README.md",
    "calystay/probe_localization": "data/probe_localization/README.md",
    "calystay/probe_structure_classifier": "data/probe_loc_struc_classifier/README.md",
    "tanyasg/scrnaseq_data": "data/scrnaseq_counts/README.md",
    "tanyasg/scrnaseq_raw": "data/scrnaseq_raw/README.md",
}


def update_all_quilt_readmes():
    for k, v in README_MAP.items():
        update_quilt_readme(pkg_name=k, new_readme_loc=v)


def make_pkg_maps(
    all_pkgs=[
        "rorydm/2d_segmented_fields",
        "rorydm/2d_segmented_fields2",
        "rorydm/2d_segmented_fields_actn2",
        "rorydm/2d_segmented_fields_actn2_2",
        "rorydm/2d_autocontrasted_fields_and_single_cells",
        "rorydm/2d_autocontrasted_fields_and_single_cells_part2",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2_2",
        "tanyasg/2d_autocontrasted_single_cell_features",
        "tanyasg/2d_autocontrasted_single_cell_features2",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2_2",
        "matheus/assay_dev_fish_analysis",
        "tanyasg/struct_scores_bonus",
        "tanyasg/struct_scores_actn2",
        "tanyasg/struct_scores_actn2_2",
        "tanyasg/struct_scores_actn2_live",
        "rorydm/manuscript_plots",
        "tanyasg/revised_manuscript_plots",
        "calystay/2d_nuclear_masks",
        "calystay/3d_actn2_segmentation",
        "calystay/probe_localization",
        "calystay/probe_structure_classifier",
        "tanyasg/2d_nonstructure_fields",
        "tanyasg/2d_nonstructure_single_cell_features",
        "tanyasg/scrnaseq_data",
        "tanyasg/scrnaseq_raw",
    ],
    skip_test_pkgs=[
        "tanyasg/scrnaseq_raw_test",
        "rorydm/manuscript_plots_test",
        "tanyasg/revised_manuscript_plots_test",
        "rorydm/2d_segmented_fields2_test",
        "rorydm/2d_segmented_fields_actn2_test",
        "rorydm/2d_segmented_fields_actn2_2_test",
        "rorydm/2d_autocontrasted_fields_and_single_cells_part2_test",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2_test",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2_2_test",
        "tanyasg/2d_autocontrasted_single_cell_features2_test",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2_test",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2_2_test",
        "tanyasg/struct_scores_bonus_test",
        "tanyasg/struct_scores_actn2_test",
        "tanyasg/struct_scores_actn2_2_test",
        "tanyasg/struct_scores_actn2_live_test",
    ],
    manual_rename={
        "rorydm/2d_segmented_fields": "2d_segmented_fields_fish_1",
        "rorydm/2d_segmented_fields2": "2d_segmented_fields_fish_2",
        "rorydm/2d_segmented_fields_actn2": "2d_segmented_fields_fish_3",
        "rorydm/2d_segmented_fields_actn2_2": "2d_segmented_fields_fish_4",
        "rorydm/2d_autocontrasted_fields_and_single_cells": "2d_autocontrasted_fields_and_single_cells_fish_1",
        "rorydm/2d_autocontrasted_fields_and_single_cells_part2": "2d_autocontrasted_fields_and_single_cells_fish_2",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2": "2d_autocontrasted_fields_and_single_cells_fish_3",
        "rorydm/2d_autocontrasted_fields_and_single_cells_actn2_2": "2d_autocontrasted_fields_and_single_cells_fish_4",
        "tanyasg/2d_autocontrasted_single_cell_features": "2d_autocontrasted_single_cell_features_fish_1",
        "tanyasg/2d_autocontrasted_single_cell_features2": "2d_autocontrasted_single_cell_features_fish_2",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2": "2d_autocontrasted_single_cell_features_fish_3",
        "tanyasg/2d_autocontrasted_single_cell_features_actn2_2": "2d_autocontrasted_single_cell_features_fish_4",
        "matheus/assay_dev_fish_analysis": "automated_local_and_global_structure_fish_1",
        "tanyasg/struct_scores_bonus": "automated_local_and_global_structure_fish_2",
        "tanyasg/struct_scores_actn2": "automated_local_and_global_structure_fish_3",
        "tanyasg/struct_scores_actn2_2": "automated_local_and_global_structure_fish_4",
        "tanyasg/struct_scores_actn2_live": "automated_local_and_global_structure_live",
    },
    internal_s3_url="s3://allencell-internal-quilt",
):

    # grab a list of packages that actually exist internally
    pkg_list = list(quilt3.list_packages(internal_s3_url))

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
    ) - len(skip_test_pkgs)

    # once checked, filter on only what's there
    pkg_map = {k: v for k, v in pkg_map.items() if k in pkg_list}
    pkg_map_test = {k: v for k, v in pkg_map_test.items() if k in pkg_list}

    return pkg_map, pkg_map_test


def aggregate_and_push(
    pkg_map,
    source_S3_url="s3://allencell-internal-quilt",
    dest_S3_url="s3://allencell",
    dest_pkg_name="aics/integrated_transcriptomics_structural_organization_hipsc_cm",
    message="Public data set",
    public=False,
):

    internal = boto3.session.Session(profile_name="default")  # noqa: F841

    # real data
    q = quilt3.Package()
    q.set("README.md", "../../data/README.md")
    q.set(
        "resources/Website_schematic_data_flow_20200310_v2.png",
        "../../data/resources/Website_schematic_data_flow_20200310_v2.png",
    )

    for (low_level_pkg_str, new_subdir) in pkg_map.items():
        p = quilt3.Package.browse(low_level_pkg_str, source_S3_url)
        for (logical_key, physical_key) in p.walk():
            q.set(f"{new_subdir}/{logical_key}", physical_key)

    git_commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    label = f"{message}. git commit hash of fish_morphology_code = {git_commit_hash}."

    # set profile to public bucket access if pushing public
    if public:
        external = boto3.session.Session(profile_name="allencell")  # noqa: F841
    #         external = boto3.session.Session(profile_name="public")  # noqa: F841

    q.push(dest_pkg_name, dest_S3_url, message=label)


PKG_MAP, PKG_MAP_TEST = make_pkg_maps()


def agg_push_test():
    aggregate_and_push(
        PKG_MAP_TEST,
        source_S3_url="s3://allencell-internal-quilt",
        dest_S3_url="s3://allencell-internal-quilt",
        dest_pkg_name="aics/integrated_transcriptomics_structural_organization_hipsc_cm_test",
        message="Public data set",
    )


def agg_push_full():
    aggregate_and_push(
        PKG_MAP,
        source_S3_url="s3://allencell-internal-quilt",
        dest_S3_url="s3://allencell-internal-quilt",
        dest_pkg_name="aics/integrated_transcriptomics_structural_organization_hipsc_cm",
        message="Public data set",
    )


def agg_push_public():
    aggregate_and_push(
        PKG_MAP,
        source_S3_url="s3://allencell-internal-quilt",
        dest_S3_url="s3://allencell",
        dest_pkg_name="aics/integrated_transcriptomics_structural_organization_hipsc_cm",
        message="Public data set",
        public=True,
    )


def main_consolidate_full():
    fire.Fire(agg_push_full)


def main_consolidate_test():
    fire.Fire(agg_push_test)
