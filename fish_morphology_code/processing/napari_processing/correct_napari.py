#!/usr/bin/env python

r"""
Correct napari hand cell segmentations errors
"""


import numpy as np
from aicsimageio import AICSImage
from scipy import ndimage
from skimage import measure


def correct_napari(image_file, area_min=5000):
    r"""
    Fix annotation errors (distinct objects labeled with same index), remove
    random small annotation scribbles, and fill holes
    Args:
        image_file (str): path to image with napari cell labels
    Returns:
        final_image (numpy.ndarray): corrected array
    """

    napari_annotation = AICSImage(image_file)
    napari_annotation_image = napari_annotation.data[0, 0, 0, 0]

    label_img = measure.label(napari_annotation_image, connectivity=1)

    napari_regions = measure.regionprops(napari_annotation_image)
    label_regions = measure.regionprops(label_img)

    napari_regions = np.array([r for r in napari_regions if r.area > area_min])
    label_regions = np.array([r for r in label_regions if r.area > area_min])

    # which images were mis-annotated
    if len(napari_regions) != len(label_regions):
        print("Fixing image: ", image_file)

    # filter small objects
    labels = [r.label for r in label_regions]

    filtered_image = np.zeros_like(
        label_img
    )  # array of 0's with same dimensions as image

    for r in labels:
        filtered_image[label_img == r] = r  # only keep size filtered labels

    # relabel filtered image
    filtered = measure.label(filtered_image)
    filtered_label_regions = measure.regionprops(filtered)
    filtered_labels = [r.label for r in filtered_label_regions]

    # fill in holes in each object in filtered image
    filtered_filled = np.zeros_like(filtered)

    for r in filtered_labels:
        current_label = np.zeros_like(filtered)
        current_label[filtered == r] = r
        filled = ndimage.binary_fill_holes(current_label)
        filtered_filled[filled] = r

    final_image = filtered_filled.astype(np.uint16)
    return final_image
