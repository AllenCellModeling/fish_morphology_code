"""
extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import numpy as np
from aicsimageio import AICSImage
import imageio
from skimage.exposure import rescale_intensity, histogram
from skimage import img_as_ubyte


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

    # convert to range 0,255 8-bit image if not already
    im_array_n = img_as_ubyte(rescale_intensity(im_array))

    # count number of nonzero pixels
    pixel_count = (im_array_n > 0).sum()

    # not sure what these are
    limit = pixel_count * upper_limit_frac
    threshold = pixel_count * low_thresh_frac

    # histogram of pixel values with bins = 0,1,2,...,255
    hist, bins = histogram(im_array_n)

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

    out_array = rescale_intensity(im_array_n, in_range=(low_thresh, high_thresh))

    if verbose:
        print("inpput array shape is {}".format(im_array.shape))
        print(
            "low_thresh = {}, high_thresh = {}, im_array_n.min()={}".format(
                low_thresh, high_thresh, np.min(im_array_n)
            )
        )
        print("out array shape is {}".format(out_array.shape))

    return out_array


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
        img_as_ubyte(rescale_intensity(im.get_image_data("ZYX", T=0, C=c).max(axis=0)))
        for c in sorted(channels.values())
    ]

    # auto contrast if image channel, original max proj if not
    Cautos = [
        auto_contrast_fn(Cdata, verbose=verbose) if c in raw_inds else Cdata
        for c, Cdata in enumerate(Cmaxs)
    ]

    return Cmaxs, Cautos


def cell_worker(
    Cautos,
    label_image,
    cell_ind,
    basename="unnamed_image_field",
    out_dir=None,
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
    verbose=True,
):

    if out_dir is None:
        out_dir = os.getcwd()

    y, x = np.where(label_image == cell_ind)
    label_crop = label_image[min(y) : max(y) + 1, min(x) : max(x) + 1]
    mask = (label_crop == cell_ind).astype(np.float64)

    for c, channel in channels.items():
        cell_object_crop = Cautos[channel][min(y) : max(y) + 1, min(x) : max(x) + 1]
        cell_object_crop = cell_object_crop * mask

        out_filename = "{0}_cell{1}_C{2}.png".format(basename, cell_ind, channel)
        out_path = os.path.join(out_dir, out_filename)
        imageio.imwrite(out_path, cell_object_crop)


def stretch_worker(
    filename,
    out_dir=None,
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
    verbose=True,
):

    if out_dir is None:
        out_dir = os.getcwd()

    basename, ext = os.path.splitext(os.path.basename(filename))

    Cmaxs, Cautos = read_and_contrast_image(filename)
    label_image = Cautos[channels["cell"]]  # extract napari annotation channel
    num_labels = np.max(label_image)
    if verbose:
        print("found {} segmented cells".format(num_labels))

    for cell_ind in range(1, num_labels + 1):
        cell_worker(
            Cautos,
            label_image,
            cell_ind,
            basename=basename,
            out_dir=out_dir,
            channels=channels,
            verbose=verbose,
        )
