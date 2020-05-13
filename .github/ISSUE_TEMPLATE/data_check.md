---
name: Dataset checklist
about: '"dataset audit tracking"'
title: 'dataset check: '
labels: 'data release'
assignees: 'donovanr'

---

Big picture:
- [ ] this dataset is on quilt with a readme
- [ ] there is a "test" version of the dataset (i.e. a couple of rows/samples) that's small / quick to download
- [ ] code to put dataset+readme is on quilt is in `/data` in a well named subdirectory 
- [ ] code to genereate data is in `fish_morphology_code/processing` in a well named subdirectory
- [ ] upstream data needed to generate this data is in quilt
- [ ] data/manifest can get merged into the main feature dataset on `["fov_path", "napariCell_ObjectNumber"]` and not lose any/too many cells (ignore if on different cells e.g. scRNAseq)
- [ ] there is a function to download the dataset in `fish_morphology_code/fish_morphology_code/bin/download_quilt_data.py`
- [ ] there is a test to test downloading the (test/small version) of the dataset in `fish_morphology_code/fish_morphology_code/tests/`


You can grab the feature dataset to test merging with
```python
p_feats = quilt3.Package.browse(
    "tanyasg/2d_autocontrasted_single_cell_features",
    "s3://allencell-internal-quilt",
)
df_feats = p_feats["features"]["a749d0e2_cp_features.csv"]()
```


Reminder that ideally code:
 - should have comments
 - should be runable by someone else, e.g. should not depend on local paths in the filesystem (other than for uploading raw data)
 - should be organized into functions that have reasonable scope, with docstrings describing inputs / outputs
 - should have tests
