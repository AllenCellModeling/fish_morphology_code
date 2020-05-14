# AICS cardio FISH probe localization features 

**Overview:** Extract probe localization features
The features were computed to study relationship between probe localization and organization ACTN structure

Images used to generate the numbers:
1. 2D nucleus segmentation
2. 2D napari annotation
3. 2D Probe segmentations
4. 3D ACTN structure classification

---------------------------------------------------------------
COLUMN NAMES								COLUMN DESCRIPTIONS:
---------------------------------------------------------------
* RawPath										Path to raw data
* fov_path									Path to raw data truncated to before `_C0.tif`
* nuc_mask_path								Path to nucleus segmentation image
* cell_num									Cell number from napari annotation
* cell_px										# px of cytoplasm (exclude nucleus)
* nuc_px										# px of nucleus
* area_px_class_#								# px of cell identified as class # structure (exclude nucleus)
* area_px_OUTSIDE_class_#						# px of cell NOT identified as class # structure
* seg_WAVELENGTH_total_probe_cyto				# px of probes in cytoplasm (exclude nucleus)
* seg_WAVELENGTH_probe_px_nuc					# px of probes in nucleus
* seg_WAVELENGTH_probe_px_class_#				# px of probes in class # structure (exclude nucleus)
* seg_WAVELENGTH_probe_px_OUTSIDE_class_#		# px of probes NOT in class # structure
* seg_WAVELENGTH_probe_density_class_#		# px of probes in class # divided by # px of class # structure
* seg_WAVELENGTH_probe_density_OUTSIDE_class_#	# of probes NOT in class # divided by # px NOT class # structure


