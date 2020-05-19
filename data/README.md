# Dataset for Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes

This data package contains the input data for all analyses in the manuscript <insert bioarxiv link here> in a compute-friendly form.
Not all of these data were used in the manuscript, but all of the data used in the manuscript are included here.

## Overview
Notable, we provide 478 fields of view containing approximately 5000 segmented single cells in different stages of cardiomyogenisis, imaged in five channels:
- Bright field
- Hoecst nuclear stain
- Endogenously GFP-tagged alpha-actinin-2 structure
- Two FISH probes per cell (eight probes overall)

Also included are FISH images of cells without a GFP labeled structure, and scRNAseq (smart-seq) data collected on approximately 40,000 cells that underwent similar differentiaion protocals as the cells we imaged.

## Access
The data is programatically accessible via `quilt`, using the code tab at the top of this page, and is also (somewhat) browseable via this web ui.
Notabley absent from this release are the raw 3D imagages from which our 2D images are derived.  These will be included shortly.

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

