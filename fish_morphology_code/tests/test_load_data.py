#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.analysis.plots import (
    load_data,
    load_main_feat_data,
    adata_manipulations,
)


def test_load_main_feat_data():
    _ = load_main_feat_data()


def test_adata_manipulations():
    df_feats = load_main_feat_data()
    _ = adata_manipulations(df_feats)


def test_load_data():
    "test loading the data"
    df, df_tidy, df_regression = load_data()
