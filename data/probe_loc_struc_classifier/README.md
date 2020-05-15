# AICS cardio FISH probe localization features 

**Overview:** Extract probe localization features
The features were computed to study relationship between probe localization and organization ACTN structure

Images used to generate the numbers:
1. 2D nucleus segmentation
2. 2D napari annotation
3. 2D Probe segmentations
4. 3D ACTN structure classification

## `Column names` and column descriptions
* `RawPath`										Path to raw data
* `fov_path`									Path to raw data truncated to before `_C0.tif`
* `nuc_mask_path`								Path to nucleus segmentation image
* `cell_num`									Cell number from napari annotation
* `cell_px`										Number of px of cytoplasm (exclude nucleus)
* `nuc_px`										Number of px of nucleus
* `area_px_class_k`								Number of px of cell identified as class k structure (exclude nucleus)
* `area_px_OUTSIDE_class_k`						Number of px of cell NOT identified as class k structure
* `seg_WAVELENGTH_total_probe_cyto`				Number of px of probes in cytoplasm (exclude nucleus)
* `seg_WAVELENGTH_probe_px_nuc`					Number of px of probes in nucleus
* `seg_WAVELENGTH_probe_px_class_k`				Number of px of probes in class k structure (exclude nucleus)
* `seg_WAVELENGTH_probe_px_OUTSIDE_class_k`		Number of px of probes NOT in class k structure
* `seg_WAVELENGTH_probe_density_class_k`		Number of px of probes in class k divided by number of px of class k structure
* `seg_WAVELENGTH_probe_density_OUTSIDE_class_k`	Number of of probes NOT in class k divided by number of px NOT class k structure
