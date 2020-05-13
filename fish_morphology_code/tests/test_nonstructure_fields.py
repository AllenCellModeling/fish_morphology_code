#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.download_quilt_data import download_nonstructure_2D_segs


def test_download():
    "test downloading the data"
    download_nonstructure_2D_segs(test=True)
