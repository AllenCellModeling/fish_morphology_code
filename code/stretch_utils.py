r"""
Extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
import imageio
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte, img_as_float64
import warnings
from tqdm import tqdm


def img_as_ubyte_nowarn(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_out = img_as_ubyte(img)
    return img_out


def rescale_intensity_nowarn(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_out = rescale_intensity(img)
    return img_out


def auto_contrast_fn(
    im_array, clip_quantiles=[0.0, 0.999], zero_below_median=False, verbose=True
):
    r"""
    im_array: 2d np.array, usually dtype=uint16
    clip_quantiles: where to set image intensity rescaling thresholds, default = clip_quantiles=[0.0,0.999]
    zero_below_median: set pixels below the median pixel value equal to zero before rescaling intensity, default=False
    verbose: print image info, default=True
    """

    im = img_as_float64(im_array)
    q_low, q_high = np.quantile(im, clip_quantiles)
    im_clipped = np.clip(im, a_min=q_low, a_max=q_high)
    if zero_below_median:
        im_clipped[im_clipped < np.median(im_clipped)] = 0
    return img_as_ubyte_nowarn(rescale_intensity_nowarn(im_clipped))


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
    fluor_channels=["488", "561", "638", "nuc"],
    bf_channels=["bf"],
    fluor_kwargs={"clip_quantiles": [0.0, 0.998], "zero_below_median": False},
    bf_kwargs={"clip_quantiles": [0.00001, 0.99999], "zero_below_median": False},
    verbose=True,
):
    r"""
    Load an image from a file path, return two lists: max projects per channel,
    and autocontrast versions of same.
    Args:
        image_path (str): location of input tiff image
        channels (dict): {"channel_name":channel_index} map for input tiff
        fluor_channels (list): list of channel names to be contrast stretched with
        bf_channels (list): list of channel names to be contrast stretched
        fluor_kwargs, default = {clip_quantiles:[0.0,0.999], zero_below_median:False}
        bf_kwargs, default = {clip_quantiles:[0.00001, 0.99999], zero_below_median:False}
        verbose (bool): print info while processing or not
    Returns:
        (Cmaxs, Cautos): tuple of two lists
            - unadjusted maxprojects per channel
            - stretched versions of same (for channels to be stretched, else unadjusted)
    """

    # channel indices on which to apply auto_contrast_fn
    fluor_inds = [ind for channel, ind in channels.items() if channel in fluor_channels]
    bf_inds = [ind for channel, ind in channels.items() if channel in bf_channels]

    # read in all data for image
    im = AICSImage(image_path)

    # list of max projects for each channels
    Cmaxs = [
        img_as_ubyte_nowarn(
            rescale_intensity_nowarn(im.get_image_data("ZYX", T=0, C=c).max(axis=0))
        )
        for c in sorted(channels.values())
    ]

    # auto contrast if image channel, original max proj if not
    Cautos = [
        auto_contrast_fn(Cdata, verbose=verbose, **fluor_kwargs)
        if c in fluor_inds
        else auto_contrast_fn(Cdata, verbose=verbose, **bf_kwargs)
        if c in bf_inds
        else Cdata
        for c, Cdata in enumerate(Cmaxs)
    ]

    return Cmaxs, Cautos


def cell_worker(
    Cautos,
    label_image,
    cell_ind,
    cell_label_value,
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

    img_out_dir = os.path.join(out_dir, "output_single_cell_images")
    os.makedirs(img_out_dir, exist_ok=True)

    y, x = np.where(label_image == cell_label_value)
    crop_slice = np.s_[min(y) : max(y) + 1, min(x) : max(x) + 1]
    label_crop = label_image[crop_slice]
    mask = (label_crop == cell_label_value).astype(np.float64)

    cell_info_df = pd.DataFrame.from_records(
        list(channels.items()), columns=["channel_name", "channel_index"]
    )
    cell_info_df["field_image_name"] = basename
    cell_info_df["cell_index"] = cell_ind
    cell_info_df["cell_label_value"] = cell_label_value
    cell_info_df["single_cell_channel_output_path"] = ""

    for i, row in cell_info_df.iterrows():
        channel, c = row["channel_name"], row["channel_index"]
        cell_object_crop = Cautos[c][crop_slice]
        cell_object_crop = cell_object_crop * mask
        cell_object_crop = img_as_ubyte_nowarn(
            rescale_intensity_nowarn(cell_object_crop)
        )

        out_filename = "{0}_cell{1}_C{2}.png".format(basename, cell_ind, channel)
        out_path = os.path.join(img_out_dir, out_filename)
        imageio.imwrite(out_path, cell_object_crop)
        cell_info_df.at[i, "single_cell_channel_output_path"] = out_path

    if verbose:
        print("cell_ind = {}, cell_label_value = {}".format(cell_ind, cell_label_value))

    return cell_info_df


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
    auto_contrast_kwargs={
        "fluor_channels": ["488", "561", "638", "nuc"],
        "bf_channels": ["bf"],
        "fluor_kwargs": {"clip_quantiles": [0.0, 0.998], "zero_below_median": False},
        "bf_kwargs": {"clip_quantiles": [0.00001, 0.99999], "zero_below_median": False},
    },
    verbose=True,
):

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    basename, ext = os.path.splitext(os.path.basename(filename))

    Cmaxs, Cautos = read_and_contrast_image(
        filename, verbose=verbose, **auto_contrast_kwargs
    )

    label_image = Cautos[channels["cell"]]  # extract napari annotation channel
    labels = np.unique(label_image)
    cell_labels = labels[labels > 0]

    if verbose:
        print("processing {}".format(filename))
        print("found {} segmented cells".format(len(cell_labels)))

    cell_info_dfs = []

    for cell_ind, cell_label_value in enumerate(tqdm(sorted(cell_labels), desc="Cell")):
        cell_info_df = cell_worker(
            Cautos,
            label_image,
            cell_ind + 1,
            cell_label_value,
            basename=basename,
            out_dir=out_dir,
            channels=channels,
            verbose=verbose,
        )

        cell_info_dfs += [cell_info_df]

    field_info_df = pd.concat(cell_info_dfs, axis="rows", ignore_index=True)
    field_info_df["field_image_path"] = filename

    return field_info_df
