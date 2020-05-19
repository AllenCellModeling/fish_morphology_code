#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.stretch import run


def test_stretch():
    """test the auto contrast and single cell segmentation code"""
    run(
        "quilt_data_test/metadata.csv",
        "quilt_data_test/supporting_files/channel_defs.json",
        out_dir="output_data_test",
    )
