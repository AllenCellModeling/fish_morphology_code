#!/usr/bin/env python
# -*- coding: utf-8 -*-

import quilt3
from fish_morphology_code.bin.download_quilt_data import download_probe_localization_features


def test_download():
    "test downloading the data"
    download_probe_localization_features(test=True)


def test_merge():
    """Check ability to merge on main df"""

    # main df
    p = quilt3.Package.browse(
        "rorydm/manuscript_plots", "s3://allencell-internal-quilt"
    )
    df = p["data.csv"]()

    # new df
    p_new = quilt3.Package.browse(
        "calystay/probe_localization", "s3://allencell-internal-quilt"
    )
    df_new = p_new["metadata.csv"]().rename(
        columns={"original_fov_location": "FOV path", "napariCell_ObjectNumber": "Cell number"}
    )
    # merge
    df_merged = df.merge(df_new, how="inner", on=["FOV path", "Cell number"])
    assert len(df_merged) == len(df)
