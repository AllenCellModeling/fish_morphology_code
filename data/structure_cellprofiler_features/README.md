# AICS cardio FISH features 

**Overview:** This dataset contains transcript levels, cell, and nuclear features from
FISH experiments on cardiomyocytes

## RNA-FISH 
RNA-FISH experiments were performed on glass replated, fixed cardiomyocytes aged to 
D18 (plate 5500000013) and D30 (plate 5500000014). All samples are the AICS-75 (clone 85)
ACTN2-mEGFP cell line, which tags the sarcomere in cardiomyocytes. Two genes were
targeted for FISH (hcr method) in each well. 

## Imaging 
FISH plates were imaged on 3i microscope and raw 3D images contain the following channels:
0: 638 (probe 2)
1: nuclear dye (dapi)
2: brightfield
3: 561 (probe 1)
4: brightfield
5: 488 (ACTN2 structure)

## Segmentation and analysis
Cells boundaries were hand drawn by three annotators. Cell segmentations are in 2D (annotators
used the 3D raw image to segment). Raw images were processed to segment the probe
and ACTN2 structure channels, and raw nuclear, brightfield, and structure fluorescent
channels were normalized before being analyzed using cellprofiler. Cell profiler was
used to calculate per cell probe counts, cell and nuclei shape features (ex. area, perimeter,
etc), and texture features from brightfield, ACTN2 structure, and nuclear normalized 
fluorescent channels. 

Nuclei were assigned to cells if their centroid overlapped the cell boundary. Multiple nuclei
assigned to one cell were merged into one object for calculation of shape and texture features.

## Merged feature csv
Each row in feature csv file correponds to one hand drawn cell. 

Merged feature csv includes the following columns:

- `ImageNumber`: image id assigned by cellprofiler
- `ImagePath`: location of images used as input to cellprofiler (full field of view)
- `ImageFailed`: TRUE indicates that image was not processed correctly by cellprofiler
- `napariCell`: cell id (this is assigned during annotation and not by cellprofiler)
- `rescaled_2D_fov_tiff_path`: location of auto-contrasted images in quilt
- `rescaled_2D_single_cell_tiff_path`: location of single cell channel images
- `fov_path`: location (basename) of raw 3i images (full field of view)
- `probe564`: probe 561
- `probe647`: probe 638
- `napariCell_nuclei_Count`: number of nuclei in each cell (calculated by cellprofiler)
- `finalnuc_border`: TRUE indicates that merged nuclei object touches border of image
- `cell_border`: TRUE indicates that napari annotated cell touches border of image
- `mh_structure_org_score` and `kg_structure_org_score`: manual sarcomere organization scores
- `probe_561_loc_score` and `probe_638_loc_score`: manual probe localization scores
- all other columns starting with `finalnuc_`: nuclear shape and texture features; only for MYH6/MYH7 probes
- all other columns starting with `napariCell_`: cell shape and texture features

### Notes on texture:
Texture features are calculated for nuclear and cell objects from 3 normalized channels:
brightfield (`_bf_`), nuclear (`_nuc_`), and ACTN2 structure (`_structure_`) at 3 scales (3, 5, 10). 

### Image object count csv
Image object count csv contains cellprofiler counts of objects in each image/fov.

Objects:
- `Count_nuc`: number of segmented nuclei b/f filtering for size
- `Count_FilterNuc`: number of segmented nuclei that pass minimum size filter (area >= 5000)
- `Count_NucBubble`: number of segmented nuclei that fail minimum size filter (area < 5000)
- `Count_FilterNucBorder`: number of size filtered nuclei that do not touch border of image
- `Count_FinalNuc`: number of merged nuclei objects (nuclei whose centroids fall within one cell are merged into one object)
- `Count_FinalNucBorder`: number of merged nuclei objects that do not touch the border of the image
- `Count_napari_cell`: number of napari annotated cells (no filters applied)
- `Count_seg_probe_561`: probe 561 counts
- `Count_seg_probe_638`: probe 638 counts

