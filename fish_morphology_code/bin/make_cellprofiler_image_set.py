#!/usr/bin/env python


import fire
import pandas as pd
import json
from fish_morphology_code.processing.cellprofiler_image_set.cp_image_set import (
    image_set_list,
)


def make_imageset(
    image_csv="",
    defaults_json="",
    path_key="2D_tiff_path",
    local_path="./",
    out_loc="image_set.csv",
):
    """
    Create and write to file image set listing (csv) that can be used as input to cellprofiler in headless mode

    Args:
        image_csv (str): location (absolute path) of csv with image locations; probably quilt metadata.csv but doesn't have to be
        defaults_json (str): location of json with default image set metadata
        path_key (str): column in image_csv with path to images
        local_path (str): path to local directory where images listed in image_csv are saved
        out_loc (str): location (absolute path) to save image set list csv
    """

    metadata_df = pd.read_csv(image_csv)

    with open(defaults_json, "r") as f:
        DEFAULT_IMAGE_SET_KWARGS = json.load(f)

    # initialize data frame that will store cellprofiler image set
    image_set_df = pd.DataFrame()

    # add images to image_set_df in format expected by cellprofiler
    all_images = metadata_df[path_key].unique()
    for i, image_path in enumerate(all_images):
        image_index = i + 1
        add_df = image_set_list(
            quilt_path=image_path,
            index=image_index,
            local_path=local_path,
            **DEFAULT_IMAGE_SET_KWARGS,
        )

        image_set_df = pd.concat([image_set_df, add_df], ignore_index=True)

    image_set_df.to_csv(out_loc, index=False)


def main_make_imageset():
    fire.Fire(make_imageset)
