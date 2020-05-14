#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.download_quilt_data import download_probe_struc_classifier_features


def test_download():
    "test downloading the data"
    download_probe_struc_classifier_features(test=True)
