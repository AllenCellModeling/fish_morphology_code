# AICS cardio RNA-FISH features 

**Overview:** This dataset contains transcript levels, cell, and nuclear features from
FISH experiments on unedited AICS-0 cardiomyocytes only (no alpha-actinin-2-mEGFP/sarcomere
structure channel)

## RNA-FISH 
FISH experiments were performed on glass re-plated, fixed cardiomyocytes aged to 
~D18 and ~D30. All samples are the unedited parental WTC line, AICS-0. We refer to this
data as non-structure RNA-FISH to differentiate it from RNA-FISH experiments done on AICS-75
(cl. 85) ACTN2-mEGFP line, which labels the sarcomeres in cardiomyocytes. We did not assess
sarcomere structure organization in the non-structure data set.
Three genes were targeted for RNA-FISH (Molecular Instruments HCR v3 method) in each well. 

## Imaging 
FISH plates were imaged on Zeiss microscopes (either ZSD2 or ZSD3) and raw 3D images contained the following channels:
0: bright field
1: 488 (probe 1)
2: 561 (probe 2)
3: 638 (probe 3)
4: nuclear dye (DAPI)

## Segmentation and analysis
Cells boundaries were hand drawn by three annotators. Cell segmentations are in 2D (annotators
used the 3D raw image to guide segmentation). Raw images were processed to segment the probes.
Cellprofiler was used to calculate per cell probe counts, cell and nuclei shape features
(ex. area, perimeter, etc), and texture features from brightfield and nuclear fluorescent channels. 

Nuclei were assigned to cells if their centroid overlapped the cell boundary. Multiple nuclei
assigned to one cell were merged into one object for calculation of shape and texture features.

## Merged feature csv
Each row in feature csv file corresponds to one hand drawn cell. 

Merged feature csv includes the following columns:

- `ImageNumber`: image id assigned by cellprofiler
- `ImagePath`: location of images used as input to cellprofiler (full field of view)
- `ImageFailed`: TRUE indicates that image was not processed correctly by cellprofiler
- `napariCell`: cell id (this is assigned during annotation and not by cellprofiler)
- `rescaled_2D_fov_tiff_path`: location of auto-contrasted images in quilt
- `rescaled_2D_single_cell_tiff_path`: location of single cell channel images
- `fov_path`: location (basename) of raw 3i images (full field of view)
- `probe488`: probe 488
- `probe564`: probe 561
- `probe647`: probe 638
- `napariCell_nuclei_Count`: number of nuclei in each cell (calculated by cellprofiler)
- `finalnuc_border`: TRUE indicates that merged nuclei object touches border of image
- `cell_border`: TRUE indicates that napari annotated cell touches border of image
- all other columns starting with `finalnuc_`: nuclear shape and texture features
- all other columns starting with `napariCell_`: cell shape and texture features

### Notes on texture:
Cellprofiler texture features are calculated for nuclear and cell objects from 2 channels:
brightfield (`_bf_`) and nuclear (`_nuc_`) at 3 scales (3, 5, 10). 

### Image object count csv
Image object count csv contains cellprofiler counts of objects in each image/fov.

Objects:
- `Count_nuc`: number of segmented nuclei b/f filtering for size
- `Count_FilterNuc`: number of segmented nuclei that pass minimum size filter (area >= 500)
- `Count_NucBubble`: number of segmented nuclei that fail minimum size filter (area < 500)
- `Count_FilterNucBorder`: number of size filtered nuclei that do not touch border of image
- `Count_FinalNuc`: number of merged nuclei objects (nuclei whose centroids fall within one cell are merged into one object)
- `Count_FinalNucBorder`: number of merged nuclei objects that do not touch the border of the image
- `Count_napari_cell`: number of napari annotated cells (no filters applied)
- `Count_seg_probe_488`: probe 488 counts
- `Count_seg_probe_561`: probe 561 counts
- `Count_seg_probe_638`: probe 638 counts

