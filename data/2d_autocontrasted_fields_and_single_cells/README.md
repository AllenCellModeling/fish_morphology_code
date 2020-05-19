# 2d autocontrasted fields and cells dataset

This dataset contains 10-channel 2D tiffs of all the image fields and segmented single cell images we use in our work, after they have been contrast adjusted for better visualization.
Note that the contrast adjustment is a very simple quantile-based algorithm, and is not intended to for use in downstream quantitative analysis.
All quantitative features used in our work are agnostic to these adjustments.

## Channels

The descriptions of each channel are in `supporting_files/channel_defs.json`:

 - the first five channels are maximum intensity projections of the original 3D images that were acquired.
 - the other channels are segmentations of the max project channels or otherwise derived from them.

## Columns

The following columns are included:

 - `fov_id`: **(metadata)** noncanonical id for the field of view (fov)
 - `original_fov_location`: **(metadata)** path on our local file system where raw microscopy data that generated the 2D images was stored
 - `2D_fov_tiff_path`: **(actual data)** path to the 10-channel 2D tiff included in this dataset
