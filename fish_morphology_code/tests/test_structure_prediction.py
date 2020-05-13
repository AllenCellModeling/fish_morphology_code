#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from fish_morphology_code.analysis.structure_prediction import (
    prep_human_score_regression_data
)

from fish_morphology_code.analysis.data_ingestion import widen_df

from fish_morphology_code.analysis.plots import (
    make_small_dataset,
    load_main_feat_data,
    adata_manipulations,
    get_global_structure,
    group_human_scores,
    rename_dict,
)


@pytest.fixture
def df_feats():
    return load_main_feat_data(use_cached=True)


def test_prep_human_score_regression_data(df_feats):
    adata = adata_manipulations(df_feats)
    assert len(adata) > 0
    df_small = make_small_dataset(adata)
    assert len(df_small) > 0
    df_gs = get_global_structure(use_cached=True)
    assert len(df_gs) > 0
    df_small = df_small.merge(df_gs)
    assert len(df_small) > 0
    df = widen_df(df_small)
    assert len(df) > 0
    df = group_human_scores(df)
    assert len(df) > 0
    df = df.rename(rename_dict, axis="columns")
    assert len(df) > 0
    df_reg = prep_human_score_regression_data(df)
    assert len(df_reg) > 0
