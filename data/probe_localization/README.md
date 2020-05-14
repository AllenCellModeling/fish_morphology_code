# AICS cardio FISH probe localization features 

**Overview:** Dataset of probe localization in a cell
The features were computed to study any localization of probes with the nucleus, cytoplasm or the structure.

Images used to generate the numbers:
1. 2D nucleus segmentation
2. 2D napari annotation
3. 2D Probe segmentations
4. 3D Structure segmentation

---------------------------------------------------------------
COLUMN NAMES						COLUMN DESCRIPTIONS:
---------------------------------------------------------------
* `cell_age`							Age of cardios
* `cell_ar`							Cell aspect ratio (min_feret_diameter/max_feret_diameter)
* `image_name`						Name of image
* `mask_name`							Mask used to define cell boundary (default: napari)
* `nuc_cell_euc_dist`					Distance (px) between the center of nucleus and center of cell
* `nuc_cell_euc_dist_norm`			Normalized distance (0-1) between the center of nucleus and center of cell
* `object_number`						Object number of cell (consistent with labels from napari annotations)
* `probe_561`							Name of probe in 561 channel
* `probe_638`							Name of probe in 638 channel
* `_abs_dist_nuc`						A list of values indicating distance (px) between the center of each probe and nearest px identified as nucleus
* `_abs_dist_struc`					A list of values indicating distance (px) between the center of probe and the nearest px identified as structure
* `_cell_abs_dist_nuc_total_c25`		25th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus	
* `_cell_abs_dist_nuc_total_c50`		50th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_nuc_total_c75`		75th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_nuc_total_mean`		Mean of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_nuc_total_median`	Median of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_nuc_total_std`		Standard deviation of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_nuc_total_sum`		Sum of distribution of distance between each pixel identified as probe and its nearest px identified as nucleus
* `_cell_abs_dist_struc_total_c25`	25th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_c50`	50th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_c75`	75th confidence interval of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_mean`	Mean of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_median`	Median of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_std`	Standard deviation of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_abs_dist_struc_total_sum`	Sum of distribution of distance between each pixel identified as probe and its nearest px identified as structure
* `_cell_dist_nuc_per_obj_c25`		25th confidence interval of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_c50`		50th confidence interval of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_c75`		75th confidence interval of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_mean`		Mean of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_median`		Median of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_std`		Standard deviation of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_cell_dist_nuc_per_obj_sum`		Sum of distribution of ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)	
* `_cell_dist_per_obj_bin100count`	Number of probe objects that have the distance transform map value between 0.75-1 (distance transform performed on napari cell object)
* `_cell_dist_per_obj_bin25count`		Number of probe objects that have the distance transform map value between 0-0.25 (distance transform performed on napari cell object)
* `_cell_dist_per_obj_bin50count`		Number of probe objects that have the distance transform map value between 0.25-0.5 (distance transform performed on napari cell object)
* `_cell_dist_per_obj_bin75count`		Number of probe objects that have the distance transform map value between 0.5-0.75 (distance transform performed on napari cell object)
* `_cell_dist_per_obj_c25`			25th confidence interval of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_per_obj_c50`			50th confidence interval of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_per_obj_c75`			75th confidence interval of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_per_obj_mean`			Mean of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_per_obj_median`			Median of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_per_obj_norm_area_bin100count`	Number of probe objects that have the distance transform map value between 0.75-1, normalized to the total area in a cell with the distance transform map value between 0.75-1
* `_cell_dist_per_obj_norm_area_bin25count`	Number of probe objects that have the distance transform map value between 0-0.25, normalized to the total area in a cell with the distance transform map value between 0-0.25
* `_cell_dist_per_obj_norm_area_bin50count` 	Number of probe objects that have the distance transform map value between 0.25-0.5, normalized to the total area in a cell with the distance transform map value between 0.25-0.5
* `_cell_dist_per_obj_norm_area_bin75count`	Number of probe objects that have the distance transform map value between 0.5-0.75, normalized to the total area in a cell with the distance transform map value between 0.5-0.75
* `_cell_dist_per_obj_norm_count_bin100count`	Number of probe objects that have the distance transform map value between 0.75-1, normalized to the total number of probe objects in a cell
* `_cell_dist_per_obj_norm_count_bin25count`	Number of probe objects that have the distance transform map value between 0-0.25, normalized to the total number of probe objects in a cell
* `_cell_dist_per_obj_norm_count_bin50count`	Number of probe objects that have the distance transform map value between 0.25-0.5, normalized to the total number of probe objects in a cell
* `_cell_dist_per_obj_norm_count_bin75count`	Number of probe objects that have the distance transform map value between 0.5-0.75, normalized to the total number of probe objects in a cell
* `_cell_dist_per_obj_std`			Standard deviation of distribution of distance transform map value of center of a probe object from a cell	
* `_cell_dist_per_obj_sum`			Sum of distribution of distance transform map value of center of a probe object from a cell
* `_cell_dist_struc_per_obj_c25`		25th confidence interval of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_c50`		50th confidence interval of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_c75`		75th confidence interval of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_mean`		Mean of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_median`	Median of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_std`		Standard deviation of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_struc_per_obj_sum`		Sum of distribution of distance transform map value of center of a probe object from segmented structure in a cell
* `_cell_dist_total_c25`				25th confidence interval of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_c50`				50th confidence interval of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_c75`				75th confidence interval of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_mean`				Mean of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_median`			Median of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_std`				Standard deviation of distribution of distance transform map value of all pixels identified as probe from a cell
* `_cell_dist_total_sum`				Sum of distribution of distance transform map value of all pixels identified as probe from a cell
* `_dist_per_obj`						A list of values indicating the distance transform map value of each probe object from a cell
* `_dist_with_nuc_cell`				A list of values indicating the ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_dist_with_struc_cell`				A list of values indicating the ratio between (the distance between the center of a probe object and its nearest px identified as structure) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
* `_probe_center_loc`					A list of tuples of (y, x) coordinates of each probe in a cell

--------------------------------------------------------------------------------
Methods
--------------------------------------------------------------------------------

1. Distance between two points:
math.sqrt((y1-y2)**2 + (x1-x2)**2)

2. Distance between the center of a probe object and its nearest px identified as nucleus
scipy.distance.cdist([(probe_y_coor, probe_x_coor)], [(nuc_y_coor, nuc_x_coor), ...], `euclidean`)

3. Distance between the center of a probe object and its nearest px identified as cell boundary
cell_inv = inverse of cell object
scipy.distance.cdist([(probe_y_coor, probe_x_coor)], [(cell_inv_y_coor, cell_inv_x_coor), ...], `euclidean`)

4. Ratio between (the distance between the center of a probe object and its nearest px identified as nucleus) and (the distance between the center of a probe object and its nearest px identified as cell boundary)
Method2/Method3 (Distance between the center of a probe object and its nearest px identified as nucleus / Distance between the center of a probe object and its nearest px identified as cell boundary)




