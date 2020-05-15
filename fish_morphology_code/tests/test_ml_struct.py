#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.download_quilt_data import download_ML_struct_scores


def test_ml_scores_download():
    "test downloading the data"
    download_ML_struct_scores(test=True)
