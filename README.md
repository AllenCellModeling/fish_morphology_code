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

## Downloading the data
The data for this project lives in a quilt package backed by a private S3 bucket (`s3://allencell-internal-quilt`).
To download it you need aws credentials to that bucket: ask [@cdw]( https://github.com/cdw ) or someone on software.
You can access all the data as normal through quilt, but this repo provides convenience functions to download the data from the command line (explained more below):

 - `download_2D_segs [--test=True]`
 - `download_2D_contrasted [--test=True]`
 - `download_2D_features [--test=True]`
 - `download_scrnaseq [--test=True]`

### Un-normalized 2D tiffs
These images are maximum intensity projections of the data as it came off the microscope, plus some segmentations masks.
The images in this dataset are 16 bits.
To get the whole dataset (~9 GB) of all 478 fields:
```
download_2D_segs
```
This will download the dataset to the `quilt_data` directory.
Use `--test=True` to download only two sample fields to the `quilt_data_test` directory.

### Normalized (autocontrasted) 2D tiffs
These images are autocontrasted versions of the above tiffs, using code in this repo, as well as single cell segmentated images of the fields using their segmentaitons masks.
The images in this dataset are 8 bits.
To get the whole dataset (~7 GB) of all 478 fields:
```
download_2D_contrasted
```
This will download the dataset to the `2d_autocontrasted_fields_and_single_cells` directory.
Use `--test=True` to download only two sample fields + single cells to the `2d_autocontrasted_fields_and_single_cells_test` directory.

### Single cell features from normalized (autocontrasted) 2D tiffs
The features in this dataset are computed on the single cell images in the 8-bit normalized (aurocontrasted) tiff dataset, using the cellprofiler pipeline in this repo.
To get the whole dataset (~100 MB) of features on singles cells from all 478 fields:
```
download_2D_features
```
This will download the dataset to the `quilt_data_features` directory.
Use `--test=True` to download only two sample fields + single cells to the `quilt_data_features_test` directory.

### Single cell RNA sequencing from cells in related conditions
Split-seq single cell transcriptomics on 21864 cells, quantifying the abundance of 38216 unique transcipts in the cells.
To get the whole dataset (~4 GB):
```
download_scrnaseq
```
This will download the dataset to the `quilt_data_scrnaseq` directory.
Use `--test=True` to download counts for only 10 cells (~6 MB) to the `quilt_data_scrnaseq_test` directory.

## Running the auto-contrasting code
To run the image normalization code, from the main repo dir:
```
contrast_and_segment quilt_data/metadata.csv quilt_data/supporting_files/channel_defs.json --out_dir=output_data
```

## Running cellprofiler to calculate single cell shape and texture features

Before running cellprofiler, download auto-contrasted images from quilt (ex. ```download_2D_contrasted --test=True```) into current working directory. 

Create an image set list in format accepted by cellprofiler's LoadData module.
```
make_cellprofiler_image_set \
    --image_csv ./quilt_data_contrasted_test/metadata.csv \
    --defaults_json fish_morphology_code/cellprofiler/cellprofiler_image_set_defaults.json \
    --path_key rescaled_2D_fov_tiff_path \
    --local_path ./quilt_data_contrasted_test \
    --out_loc ./test_image_set_list.csv
```

Run cellprofiler pipeline in this repository using test image set list as input:

```
#!/bin/bash
mkdir cp_out

cellprofiler \
    -p fish_morphology_code/cellprofiler/cp_3i_image_processing.cppipe \
    --run-headless \
    --data-file ./test_image_set_list.csv \
    -o ./cp_out \
    -L 10
```

To run cellprofiler on slurm, first:
```
module load anaconda3
source activate cellprofiler-3.1.8
```

## Merge single cell features calculated by cellprofiler and image metadata
To merge features, also need these files from repo:
1. **fov metadata:** ```data/input_segs_and_tiffs/labkey_fov_metadata.csv```)
2. **structure scores with fov id:** ```data/structure_scores_fov.csv```
```
merge_cellprofiler_output \
    --cp_csv_dir cp_out \
    --csv_prefix napari_ \
    --out_csv cp_out/cp_features.csv \
    --normalized_image_manifest quilt_data_contrasted_test/metadata.csv \
    --fov_metadata fish_morphology_code/data/input_segs_and_tiffs/labkey_fov_metadata.csv \
    --structure_scores fish_morphology_code/data/structure_scores_fov.csv \
    --prepend_local ./quilt_data_contrasted \
    --relative_columns "rescaled_2D_fov_tiff_path,rescaled_2D_single_cell_tiff_path"

```

## Running the tests
To run the `pytest` tests defined in `fish_morphology_code/tests` via `tox`, use
```
make build
```
This will take a while the first time setting up the test environment.

## Uploading data to quilt
Uploading to quilt should happen infrequently so all the code has been put in the `data` directory outside of the main `fish_morphology_code` library.
All uploading can be down in the `data` directory using `python distribute_<your_dataset>.py`

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


