#!/usr/bin/env python
"""
TG cleaned up script Calysta and Melissa used to extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import numpy as np
from aicsimageio import AICSImage
from skimage import exposure
import argparse
import json
import imageio


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

# Auto contrast function from Aditya
def auto_contrast_fn(im_array):
    print("in array is " + str(im_array.shape))

    # convert to range 0,255 if not already (it is already) and convert to 8-bit
    im_array_n = (im_array / im_array.max() * 255).astype("uint8")

    # count number of nonzero pixels
    im_flat = im_array_n.flatten()
    t = im_flat > 0
    im_flat[t] = 1
    pixel_count = im_flat.sum()

    # not sure what these are
    limit = pixel_count / 10
    threshold = pixel_count / 5000

    # histogram of pixel values with bin boundaries = 0,1,2,...256
    hist = np.histogram(im_array_n, np.array([i for i in range(257)]))

    low_thresh = None
    high_thresh = None

    for i in range(hist[0].size):  # iterate through all pixel values 0,1,2,...255
        if (
            hist[0][i] >= limit
        ):  # if more pixels have this value than limit, set to zero
            pos = im_array_n == i
            im_array_n[
                pos
            ] = (
                0
            )  # assumes??? pix values with high counts are low value pixels eg 0,1,2,3
        elif limit > hist[0][i] > threshold:
            if (
                low_thresh is None
            ):  # lowest pixel value with counts between limit & threshold
                low_thresh = i
            else:
                pass
        else:
            pass

    for i in range(hist[0].size):
        if limit < hist[0][hist[0].size - i - 1] < threshold:
            pos = im_array_n == i
            im_array_n[pos] = 0
        elif hist[0][hist[0].size - i - 1] >= threshold:
            high_thresh = (
                hist[0].size - i - 1
            )  # highest pixel value between limit & threshold
            break
        else:
            pass
    print(low_thresh, high_thresh, np.min(im_array_n))
    out_array = exposure.rescale_intensity(
        im_array_n, in_range=(low_thresh, high_thresh)
    )  # scale to 0,255
    print("out array is " + str(out_array.shape))
    return out_array


def read_and_contrast_image(filebase):
    Cautos = []
    Cmaxs = []
    print(filebase)
    C_im = AICSImage(filebase)
    C_data = C_im.data
    for k in range(5):

        C = C_data[0, k, :, :, :]
        print(C.shape)
        Cmax = np.amax(C, axis=0)
        print(Cmax.shape)
        Cmax_n = Cmax / Cmax.max() * 255
        print(Cmax_n.shape)
        Cmax_n_auto = auto_contrast_fn(Cmax_n)
        Cmaxs.append(Cmax)
        Cautos.append(Cmax_n_auto)
    for add in range(5, 10):
        Cmaxs.append(C_data[0, add, 0, :, :])
        Cautos.append((C_data[0, add, 0, :, :]))
    return Cmaxs, Cautos


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
