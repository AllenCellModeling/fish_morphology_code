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


@pytest.mark.incremental
class TestLoadData:
    def test_load_main_feat_data(self):
        self.df_feats = load_main_feat_data()

    def test_adata_manipulations(self):
        self.adata = adata_manipulations(self.df_feats)

    def test_make_small_dataset(self):
        self.df_small = make_small_dataset(self.adata)

    def test_get_global_structure(self):
        self.df_gs = get_global_structure()

    def test_group_human_scores(self):
        self.df_small = self.df_small.merge(self.df_gs)
        self.df = widen_df(self.df_small)
        self.df = group_human_scores(self.df)


# def test_load_data():
#     "test loading the data"
#     df, df_tidy, df_regression = load_data()
