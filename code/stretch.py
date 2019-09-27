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


def auto_contrast_fn(
    im_array,
    upper_limit_frac=1 / 10,
    low_thresh_frac=1 / 5000,
    zero_high_count_pix=True,
    verbose=True,
):
    """
    im_array: 2d np.array, usually dtype=uint16
    upper_limit_frac: the highest pixel value (in 8-bits) exceeding this fraction of all image pixels is used as the lower threshold for clipping the image, default=1/10
    low_thresh_frac: the lowest pixel value (in 8-bits) exceeding this fraction of all image pixels is used as the upper threshold for clipping the image, default=1/5000
    zero_high_count_pix: set all pixel values below (inclusive) the upper_limit_frac are set to zero, default=True
    verbose: print image info, default=True
    """

    # TODO don't convert to 8-bit and then stretch, stretch original image to get better sampling

    # TODO use better conversion function
    # convert to range 0,255 8-bit image if not already
    im_array_n = (im_array / im_array.max() * 255).astype("uint8")

    # count number of nonzero pixels
    pixel_count = (im_array_n > 0).sum()

    # not sure what these are
    limit = pixel_count * upper_limit_frac
    threshold = pixel_count * low_thresh_frac

    # histogram of pixel values with bin boundaries = 0,1,2,...256
    hist, _ = np.histogram(im_array_n, bins=np.arange(256 + 1))

    # high and low thresholds at which to constrast stretch the image
    low_thresh = np.where(hist < limit)[0].min()
    high_thresh = np.where(hist > threshold)[0].max()

    # zero out pixels values with high counts -- presumes all high counts are low values?
    if zero_high_count_pix:
        high_count_pix = np.where(hist >= limit)[0]
        high_count_mask = np.isin(im_array_n, high_count_pix)
        im_array_n[high_count_mask] = 0

    out_array = exposure.rescale_intensity(
        im_array_n, in_range=(low_thresh, high_thresh)
    )

    if verbose:
        print("inpput array shape is {}".format(im_array.shape))
        print(
            "low_thresh = {}, high_thresh = {}, im_array_n.min()={}".format(
                low_thresh, high_thresh, np.min(im_array_n)
            )
        )
        print("out array shape is {}".format(out_array.shape))

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
