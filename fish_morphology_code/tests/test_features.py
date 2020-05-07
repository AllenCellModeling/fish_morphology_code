#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.bin.download_quilt_data import download_2D_features, download_ML_struct_scores


def test_download():
    "test downloading the data"
    download_2D_features(test=True)

def test_ml_scores_download()
	"test downloading the data"
    download_ML_struct_scores(test=True)
