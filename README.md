# Fish Morphology Code

[![Build Status](https://github.com/AllenCellModeling/fish_morphology_code/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/fish_morphology_code/actions)

data ingestion, processing, and analysis for cardio/FISH project

---


## Installation
Recommended installation procedure is to create a conda environment and then pip install into that environment.

### Normal users
Clone this repository, then
```
cd fish_morphology_code
conda create --name cardio_fish_data python=3.7
conda activate cardio_fish_data
pip install -e .
```

### Developers

After following the "Normal users" installation instructions,
```
pip install -e .[all]
pre-commit install
```
This will install the the dev dependencies.
It will also install the black formatter and flake8 linter, and configure them to run as pre-commit hooks.

### Versions
**Stable Release:** `pip install fish_morphology_code`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/fish_morphology_code.git`

## Documentation
For full package documentation please visit [AllenCellModeling.github.io/fish_morphology_code](https://AllenCellModeling.github.io/fish_morphology_code).

## Downloading the data

### Un-normalized 2D tiffs
The images in this dataset are 16 bits.

#### Test
To download a test dataset (~40 MB) of just two fields:
```
download_2D_segs --test=True
```
This will download the dataset and save it to the `quilt_data_test` directory.

#### Whole dataset
To get the whole dataset (~9 GB) of all 478 fields:
```
download_2D_segs
```
This will download the dataset and save it to the `quilt_data` directory.

### Normalized (autocontrasted) 2D tiffs
The images in this dataset are 8 bits.

#### Test
To download a test dataset (~20 MB) of just two fields:
```
download_2D_contrasted --test=True
```
This will download the dataset and save it to the `2d_autocontrasted_fields_and_single_cells_test` directory.

#### Whole dataset
To get the whole dataset (~7 GB) of all 478 fields:
```
download_2D_contrasted
```
This will download the dataset and save it to the `2d_autocontrasted_fields_and_single_cells` directory.

## Running the auto-contrasting code
To run the image normalization code, from the main repo dir:
```
contrast_and_segment quilt_data/metadata.csv quilt_data/supporting_files/channel_defs.json --out_dir=output_data
```

## Running the tests
To run the `pytest` tests defined in `fish_morphology_code/tests` via `tox`, use
```
make build
```
This will take a while the first time setting up the test environment.

## Uploading data to quilt
Uploading to quilt should happen infrequently so all the code has been put in the `data` directory outside of the main `fish_morphology_code` library.

### Un-normalized 2D tiffs
To upload a new version of the un-normalized 2D fovs, from the `data` directory:
```
python distribute_seg_dataset.py
```

### Normalized 2D tiffs
To upload a new version of the normalized (autocontrasted) 2D fovs + single cell images, from the `data` directory:
```
python distribute_autocontrasted_dataset.py
```


## Bare minimum contents:

- [x] manifest of the locations of original image data + segmentations
- [x] script for taking original images and generating 2D normalized images
- [x] structure annotations for each segmented cell
- [x] text-ish version of cellprofiler workflow to generate single cell segmentations and features from image data + segmentations


## Stretch goals

- [ ] original image data + annotations into fms/labkey/quilt, or somehow programatically downloadable
