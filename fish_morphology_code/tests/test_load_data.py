#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from fish_morphology_code.analysis.data_ingestion import widen_df

from fish_morphology_code.analysis.plots import (
    make_small_dataset,
    # load_data,
    load_main_feat_data,
    adata_manipulations,
    get_global_structure,
    group_human_scores,
)


@pytest.fixture
def df_feats():
    return load_main_feat_data(use_cached=True)


def test_load_main_feat_data(df_feats):
    assert df_feats.shape[0] > 1000
    assert df_feats.shape[1] > 10


def test_adata_manipulations(df_feats):
    _ = adata_manipulations(df_feats, use_cached=True)


def test_make_small_dataset(df_feats):
    adata = adata_manipulations(df_feats)
    _ = make_small_dataset(adata, use_cached=True)


def test_get_global_structure(df_feats):
    _ = get_global_structure(use_cached=True)


def test_group_human_scores(df_feats):
    adata = adata_manipulations(df_feats, use_cached=True)
    df_small = make_small_dataset(adata, use_cached=True)
    df_gs = get_global_structure(use_cached=True)
    df_small = df_small.merge(df_gs, use_cached=True)
    df = widen_df(use_cached=True)
    _ = group_human_scores(df, use_cached=True)


# def test_load_data():
#     "test loading the data"
#     df, df_tidy, df_regression = load_data()
