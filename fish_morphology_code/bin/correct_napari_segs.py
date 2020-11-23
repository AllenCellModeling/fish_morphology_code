#!/usr/bin/env python

import os
import fire
import pandas as pd
from skimage.io import imsave

from fish_morphology_code.processing.napari_processing.correct_napari import (
    correct_napari,
)


def run_napari_correct(image_csv="annotations_file.csv", save_dir="./", area_min=5000):
    """
    Run correction on napari hand cell segmentations to fix labeling errors, remove scribbles, and fill holes
    Args:
        image_csv (str): path to csv with list of images to correct
        save_dir (str): path to directory where to save corrected images
        area_min (int): minimum cell area; napari segmentations smaller than this are filtered out
    """

    image_df = pd.read_csv(image_csv, header=0)

    for f in image_df["fov_path"].tolist():
        corrected_image = correct_napari(f)
        imsave(
            "{}/corrected_{}".format(save_dir, os.path.basename(f)),
            corrected_image,
            plugin="tifffile",
            photometric="minisblack",
        )


def main_napari_correct():
    fire.Fire(run_napari_correct)
