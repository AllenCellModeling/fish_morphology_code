#!/usr/bin/env python


import fire
import pandas as pd
import json
from fish_morphology_code.processing.cellprofiler_image_set.cp_image_set import (
    image_set_list,
    image_set_list_nonstructure,
)


def make_imageset(
    image_csv="",
    defaults_json="",
    path_key="2D_tiff_path",
    local_path="./",
    out_loc="image_set.csv",
    fish_type="",
):
    """
    Create and write to file image set listing (csv) that can be used as input to cellprofiler in headless mode

    Args:
        image_csv (str): location (absolute path) of csv with image locations; probably quilt metadata.csv but doesn't have to be
        defaults_json (str): location of json with default image set metadata
        path_key (str): column in image_csv with path to images
        local_path (str): path to local directory where images listed in image_csv are saved
        out_loc (str): location (absolute path) to save image set list csv
        fish_type (str): type of fish image; indicates whether 488 channel is "structure" (ACTN2-mEGFP) or "nonstructure" (hcr probe)
    """

    metadata_df = pd.read_csv(image_csv)

    with open(defaults_json, "r") as f:
        DEFAULT_IMAGE_SET_KWARGS = json.load(f)

    # initialize data frame that will store cellprofiler image set
    image_set_df = pd.DataFrame()

    # use the correct image set list function
    if fish_type == "structure":
        image_set_func = image_set_list
    elif fish_type == "nonstructure":
        image_set_func = image_set_list_nonstructure
    else:
        raise ValueError(
            "Undefined fish_type; allowed values are 'structure' and 'nonstructure'"
        )

    # add images to image_set_df in format expected by cellprofiler
    all_images = metadata_df[path_key].unique()
    for i, image_path in enumerate(all_images):
        image_index = i + 1
        add_df = image_set_func(
            quilt_path=image_path,
            index=image_index,
            local_path=local_path,
            **DEFAULT_IMAGE_SET_KWARGS,
        )

        # add image only if it exists
        if not add_df.empty:
            image_set_df = pd.concat([image_set_df, add_df], ignore_index=True)
        else:
            continue

    image_set_df.to_csv(out_loc, index=False)


def main_make_imageset():
    fire.Fire(make_imageset)
