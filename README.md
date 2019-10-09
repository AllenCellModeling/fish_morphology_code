# fish morphology code

basic code and automated workflows to produce analysis friendly data from annotated fish images

## Installation

In this directory:
```
pip install -r requirements.txt
```

## Running the code

One day...

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
