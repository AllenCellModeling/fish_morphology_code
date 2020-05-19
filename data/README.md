# Dataset for Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes

This data package contains the input data for all analyses in the manuscript <insert bioarxiv link here> in a compute-friendly form.
Not all of these data were used in the manuscript, but all of the data used in the manuscript are included here.

## Overview
Notably, we provide 478 fields of view containing approximately 5000 segmented single cells in different stages of cardiomyogenisis, imaged in five channels:
- Bright field
- Hoechst nuclear stain
- Endogenously GFP-tagged alpha-actinin-2 structure
- Two FISH probes per cell (eight probes overall)

Also included are
- FISH images of cells without a GFP labeled structure (~30 probes)
- scRNAseq (Split-seq) data collected on approximately 22,000 cells that underwent similar differentiaion protocals as the cells we imaged

## Organization
The data in this package is organized into seperate datasets, reflecting different data of different types (scRNAseq vs FISH / image data), and different downstream processing / feature derivation.

The datasets included in this package are:

- `2d_autocontrasted_fields_and_single_cells`
- `2d_autocontrasted_single_cell_features`
- `2d_nonstructure_fields`
- `2d_nonstructure_single_cell_features`
- `2d_nuclear_masks`
- `2d_segmented_fields`
- `3d_actn2_segmentation`
- `automated_local_and_global_structure`
- `manuscript_plots`
- `probe_localization`
- `probe_structure_classifier`
- `scrnaseq_data`
- `scrnaseq_raw`
- `segmented_nuc_labels`

Notably absent from this release are the raw 3D imagages from which our 2D images are derived.
These will be included shortly.

## Access
The data is programatically accessible via `quilt`, and is also (somewhat) browseable via this web ui.

### Bulk download
To download the entire data set, install the `quilt` python package using
```
pip install quilt
```
and then
```python
import quilt3
b = quilt3.Bucket("s3://allencell-internal-quilt")
b.fetch("aics/cardio_diff_manuscript/", "./")
```

### Download specific files or datasets
To download only certain individual files, naviagte the web ui here to the specific file you are interested in, and use the `DOWNLOAD FILE` button in the upper right of the page

To download specific folders/directories of data, similarly use the web ui to fin dthe directory you want, and check the `<> CODE` tab at the top of the page for the python code that downloads that specific subset of data.

### Programatic access
To access the data via the python quilt API, isnall `quilt` via `pip`, and then load the package with:

```python
pkg = quilt3.Package.browse(
    "aics/cardio_diff_manuscript",
    "s3://allencell-internal-quilt",
)
```
Instructions for interacting with quilt packages can be found [here](https://docs.quiltdata.com/walkthrough/getting-data-from-a-package)
