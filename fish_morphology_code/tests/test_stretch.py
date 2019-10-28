#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.stretch import run
from fish_morphology_code.bin.download_quilt_data import download_2D_segs


def test_stretch():
    """test the auto contrast and single cell segmentation code"""

    download_2D_segs(test=True)
    run(
        "quilt_data_test/metadata.csv",
        "quilt_data_test/supporting_files/channel_defs.json",
        out_dir="output_data",
    )
