# Fish Morphology Code

[![Build Status](https://github.com/AllenCellModeling/fish_morphology_code/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/fish_morphology_code/actions)

Data ingestion, processing, and analysis code for _Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes_ `<insert bioarxiv link here>`

---

## Installation
The recommended installation procedure is to create a conda environment and then pip install into that environment.

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
See the [open data release](https://open.quiltdata.com/b/allencell/tree/aics/integrated_transcriptomics_structural_organization_hipsc_cm/
) (produced by this code) for instructions to download the data:

## Notes on using the code

### Running the auto-contrasting code
To run the image normalization code, from the main repo dir:
```
contrast_and_segment quilt_data/metadata.csv quilt_data/supporting_files/channel_defs.json --out_dir=output_data
```

### Running cellprofiler to calculate single cell shape and texture features

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

### Merge single cell features calculated by cellprofiler and image metadata
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

### Running the tests
To run the `pytest` tests defined in `fish_morphology_code/tests` via `tox`, use
```
make build
```
This will take a while the first time setting up the test environment.

### Uploading data to quilt
Uploading to quilt should happen infrequently, so all the code has been put in the `data` directory outside of the main `fish_morphology_code` library.
All uploading can be done in the `data` directory using `python distribute_<your_dataset>.py`.
