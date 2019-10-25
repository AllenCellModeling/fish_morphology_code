#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

from fish_morphology_code.bin.stretch import run


def test_stretch():
    """test the auto contrast and single cell segmentation code"""
    run(
        "quilt_data_test/metadata.csv",
        "quilt_data_test/supporting_files/channel_defs.json",
    )
