#!/usr/bin/env python
"""
extract hand segmented images of individual cells for scoring actn2 structure
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
    im_array_n = (im_array / im_array.max() * np.iinfo(np.uint8).max).astype(np.uint8)

    # count number of nonzero pixels
    pixel_count = (im_array_n > 0).sum()

    # not sure what these are
    limit = pixel_count * upper_limit_frac
    threshold = pixel_count * low_thresh_frac

    # histogram of pixel values with bin boundaries = 0,1,2,...256
    hist, bins = np.histogram(
        im_array_n, bins=np.arange(1 + np.iinfo(np.uint8).max + 1)
    )

    # high and low thresholds at which to constrast stretch the image
    low_thresh = bins[0] if hist.min() >= limit else np.where(hist < limit)[0].min()
    high_thresh = (
        bins[-1] if hist.max() <= threshold else np.where(hist > threshold)[0].max()
    )

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


def normalize_image_zero_one(im):
    r"""
    Normalize a Numpy array to have min zero and max one.
    Args:
        im (numpy.ndarray): data matrix
    Returns:
        (numpy.ndarray): normalized data matrix
    """
    im = im - np.min(im)
    if np.max(im) > 0:
        im = im / np.max(im)
    return im


def float_to_uint(im, uint_dtype=np.uint8):
    r"""
    Convert an array of floats to unsigned ints, contrast stretrching so to the dynamic range of the output data type.
    Args:
        im (numpy.ndarray): data matrix
        uint_dtype (numpy.dtype): numpy data type e.g. np.uint8
    Returns:
        (numpy.ndarray): integer data matrix
    """
    imax = np.iinfo(uint_dtype).max + 1  # eg imax = 256 for uint8
    im = im * imax
    im[im == imax] = imax - 1
    im = np.asarray(im, uint_dtype)
    return im


def read_and_contrast_image(
    image_path,
    channels={
        "bf": 0,
        "488": 1,
        "561": 2,
        "638": 3,
        "nuc": 4,
        "seg488": 5,
        "seg561": 6,
        "seg638": 7,
        "backmask": 8,
        "cell": 9,
    },
    stretch_channels=["bf", "488", "561", "638", "nuc"],
    verbose=True,
):
    r"""
    Load an image from a file path, return two lists: max projects per channel,
    and autocontrast versions of same.
    Args:
        image_path (str): location of input tiff image
        channels (dict): {"channel_name":channel_index} map for input tiff
        stretch_channels (list): list of channel names that to be contrast stretched
        verbose (bool): print info while processing or not
    Returns:
        (Cmaxs, Cautos): tuple of two lists
            - unadjusted maxprojects per channel
            - stretched versions of same (for channels to be stretched, else unadjusted)
    """
    # print file path if desired
    if verbose:
        print(image_path)

    # channel indices on which to apply auto_contrast_fn
    raw_inds = [ind for channel, ind in channels.items() if channel in stretch_channels]

    # read in all data for image
    im = AICSImage(image_path)

    # list of max projects for each channels
    Cmaxs = [
        float_to_uint(
            normalize_image_zero_one(im.get_image_data("ZYX", T=0, C=c).max(axis=0)),
            uint_dtype=np.uint8,
        )
        for c in sorted(channels.values())
    ]

    # auto contrast if image channel, original max proj if not
    Cautos = [
        auto_contrast_fn(Cdata, verbose=verbose) if c in raw_inds else Cdata
        for c, Cdata in enumerate(Cmaxs)
    ]

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
