#!/usr/bin/env python
"""
extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import numpy as np
import argparse
import json
import imageio

from stretch_utils import read_and_contrast_image


parser = argparse.ArgumentParser(
    description="Extract segmented objects from field of view and save channels into separate images"
)
parser.add_argument(
    "image_files", type=str, help="csv file with list of mips with seg files"
)
parser.add_argument("channels", type=str, help="json file with channel identifiers")
parser.add_argument(
    "--out_dir",
    dest="out_dir",
    type=str,
    default=os.getcwd(),
    help="output directory where to save images",
)
args = parser.parse_args()


with open(args.image_files, "r") as images:
    file_names = [f.strip() for f in images]

with open(args.channels, "r") as f:
    channels = json.load(f)

for filename in file_names:
    basename = os.path.basename(filename)
    basename = basename.split(".")[0]

    Cmaxs, Cautos = read_and_contrast_image(filename)  # autocontrast all images, good??
    label_image = Cautos[channels["cell"]]  # extract napari annotation channel
    num_labels = np.max(label_image)
    print(num_labels)

    # extract each cell object by channel
    for cell in range(1, num_labels + 1):
        y, x = np.where(label_image == cell)
        label_crop = label_image[
            min(y) : max(y), min(x) : max(x)
        ]  # add one to max of slices?

        mask = np.zeros(label_crop.shape)
        mask[label_crop == cell] = 1

        for c in channels.keys():
            current_channel = channels[c]
            cell_object_crop = Cautos[current_channel][min(y) : max(y), min(x) : max(x)]
            cell_object_crop = cell_object_crop * mask

            imageio.imwrite(
                os.path.join(
                    args.out_dir,
                    basename
                    + "_cell"
                    + str(cell)
                    + "_C"
                    + str(current_channel)
                    + ".png",
                ),
                cell_object_crop,
            )
