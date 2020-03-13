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
    return load_main_feat_data()


def test_load_main_feat_data(df_feats):
    assert df_feats.shape[0] > 1000
    assert df_feats.shape[1] > 10


def test_adata_manipulations():
    _ = adata_manipulations(df_feats)


def test_make_small_dataset():
    adata = adata_manipulations(df_feats)
    _ = make_small_dataset(adata)


def test_get_global_structure():
    _ = get_global_structure()


def test_group_human_scores():
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    df_gs = get_global_structure()
    df_small = df_small.merge(df_gs)
    df = widen_df(df_small)
    _ = group_human_scores(df)


# def test_load_data():
#     "test loading the data"
#     df, df_tidy, df_regression = load_data()
