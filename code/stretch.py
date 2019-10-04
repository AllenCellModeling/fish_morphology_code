#!/usr/bin/env python
"""
Extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import json
import fire

from stretch_utils import stretch_worker


def run(image_file_csv, channels_json, out_dir=None):
    """
    Extract segmented objects from field of view and save channels into separate images

    image_file_csv: csv file with list of mips with seg files
    channels_json: json file with channel identifiers
    out_dir: output directory where to save images, default=current working directory
    """

    with open(image_file_csv, "r") as images:
        file_names = [f.strip() for f in images]
    with open(channels_json, "r") as f:
        channels = json.load(f)
    if out_dir is None:
        out_dir = os.getcwd()

    for filename in file_names:
        stretch_worker(filename, out_dir=out_dir, channels=channels, verbose=True)


if __name__ == "__main__":
    fire.Fire(run)
