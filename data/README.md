# 2d segmented fields dataset

This dataset contains 10-channel 2D tiffs of all the image fileds we use in our work.

## Channels

The descriptions of each channel are in `supporting_files/channel_defs.json`:

 - the first five channels are maximun intesity projections of the original 3D images that were aquired.
 - the other channels are segmentations of the max project channels or otherwise derived from them.

## Columns

The folllowing columns are included:

 - `fov_id`: **(metadata)** noncanonical id for the field of view (fov)
 - `original_fov_location`: **(metadata)** path on our local file system where raw microscopy data that generated the 2D images was stored
 - `2D_fov_tiff_path`: **(actual data)** path to the 10-channel 2D tiff included in this dataset
