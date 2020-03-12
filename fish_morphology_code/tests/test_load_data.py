#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fish_morphology_code.analysis.plots import load_data


def test_load_data():
    "test loading the data"
    df, df_tidy, df_regression_info = load_data()
