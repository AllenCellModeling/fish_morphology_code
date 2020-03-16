#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from fish_morphology_code.analysis.data_ingestion import widen_df, tidy_df

from fish_morphology_code.analysis.plots import (
    make_small_dataset,
    # load_data,
    load_main_feat_data,
    adata_manipulations,
    get_global_structure,
    group_human_scores,
    make_regression_df,
    clean_probe_names,
    add_densities,
    rename_dict,
)


@pytest.fixture
def df_feats():
    return load_main_feat_data(use_cached=True)


def test_load_main_feat_data(df_feats):
    assert df_feats.shape[0] > 1000
    assert df_feats.shape[1] > 10


def test_adata_manipulations(df_feats):
    adata = adata_manipulations(df_feats)
    assert len(adata) > 0


def test_make_small_dataset(df_feats):
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    assert len(df_small) > 0


def test_get_global_structure(df_feats):
    df_gs = get_global_structure(use_cached=True)
    assert len(df_gs) > 0


def test_group_human_scores(df_feats):
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    df_gs = get_global_structure(use_cached=True)
    df_small = df_small.merge(df_gs)
    df = widen_df(df_small)
    df = group_human_scores(df)
    assert len(df) > 0


def test_make_regression_df(df_feats):
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    df_gs = get_global_structure(use_cached=True)
    df_small = df_small.merge(df_gs)
    df = widen_df(df_small)
    df = group_human_scores(df)
    df = df.rename(rename_dict, axis="columns")
    df, df_regression_info = make_regression_df
    assert len(df) > 0
    assert len(df_regression_info) > 0


def test_clean_probe_names(df_feats):
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    df_gs = get_global_structure(use_cached=True)
    df_small = df_small.merge(df_gs)
    df = widen_df(df_small)
    df = group_human_scores(df)
    df = df.rename(rename_dict, axis="columns")
    df, df_regression_info = make_regression_df
    df_tidy = tidy_df(df)
    df = df.rename(rename_dict, axis="columns")
    df_tidy = df_tidy.rename(rename_dict, axis="columns")
    df, df_tidy = clean_probe_names(df, df_tidy)
    assert len(df) > 0
    assert len(df_tidy) > 0


def test_add_densities(df_feats):
    adata = adata_manipulations(df_feats)
    df_small = make_small_dataset(adata)
    df_gs = get_global_structure(use_cached=True)
    df_small = df_small.merge(df_gs)
    df = widen_df(df_small)
    df = group_human_scores(df)
    df = df.rename(rename_dict, axis="columns")
    df, df_regression_info = make_regression_df
    df_tidy = tidy_df(df)
    df = df.rename(rename_dict, axis="columns")
    df_tidy = df_tidy.rename(rename_dict, axis="columns")
    df, df_tidy = clean_probe_names(df, df_tidy)
    df, df_tidy = add_densities(df, df_tidy)
    assert len(df) > 0
    assert len(df_tidy) > 0


# def test_load_data():
#     "test loading the data"
#     df, df_tidy, df_regression = load_data()
