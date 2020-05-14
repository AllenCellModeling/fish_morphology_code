#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.download_quilt_data import download_2D_features


def test_download():
    "test downloading the data"
    download_2D_features(test=True)
