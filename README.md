# fish morphology code

basic code and automated workflows to produce analysis friendly data from annotated fish images

## Installation

Recommended installation procedure is to create a conda environment and then pip install into that environment.

### Normal users

Clone this repository, then

```
cd fish_morphology_code
conda create --name cardio_fish_data python=3.7
conda activate cardio_fish_data
pip install -r requirements.txt
```

### Developers

After following the "Normal users" installation instructions,

```
pip install pre-commit
pre-commit install
```

This will install the black formatter and flake8 linter, and configure them to run as pre-commit hooks.

## Contributions

Internal contributions are welcome and encouraged.
Please create an issue describing your contributions, and work an a branch with a descriptive name, such as `FEATURE/your_contribution` or `BUGFIX/the_bug` etc.
When things look good on your branch, create a pull request to master.
Direct commits to master should be disabled, but if not please don't push commits directly to the master branch.

## Running the code

To run the image normalization code, from the main repo dir:
```
python code/stretch.py data/input_segs_and_tiffs/raw_seg_013_014_images.csv data/channel_defs.json --out_dir=/allen/aics/modeling/data/cardio_fish/normalized_2D_tiffs
```

## Running the tests

### Image normalization

To run with mostly default settings, this should work (only with access to `/allen/aics`):
```
python code/stretch.py test/input_data/input_tiffs.csv test/input_data/channel_defs.json --out_dir=test/output_data/
```

## Bare minimum contents contains:

- [ ] manifest of the locations of original image data + segmentations
- [ ] script for taking original images and generating 2D normalized images
- [x] structure annotations for each segmented cell
- [ ] text-ish version of cellprofiler workflow to generate single cell segmentations and features from image data + segmentations


## Stretch goals

- [ ] original image data + annotations into fms/labkey/quilt, or somehow programatically downloadable
