r"""
Extract hand segmented images of individual cells for scoring actn2 structure
"""

import warnings
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import imageio
from skimage.exposure import rescale_intensity, histogram
from skimage import img_as_ubyte, img_as_float64

from aicsimageio import AICSImage


DEFAULT_CHANNELS = {
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
}


DEFAULT_CHANNEL_GROUPS = {
    "fluor": ["488", "561", "638", "nuc"],
    "bf": ["bf"],
    "seg": ["seg488", "seg561", "seg638", "backmask", "cell"],
}


DEFAULT_CONTRAST_KWARGS = {
    "fluor": {"clip_quantiles": [0.0, 0.998], "zero_below_median": False},
    "bf": {"clip_quantiles": [0.00001, 0.99999], "zero_below_median": False},
    "seg": {},
}


def img_as_ubyte_nowarn(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_out = img_as_ubyte(img)
    return img_out


def _simple_quantile_constrast(
    im_array,
    clip_quantiles=[0.0, 0.999],
    zero_below_median=False,
    verbose=False,
    **kwargs
):
    r"""
    Nonparametrically stretch the contrast of single channel image
        im_array (np.array): 2D array, usually dtype=uint16
        clip_quantiles (list): where to set image intensity rescaling thresholds, default=clip_quantiles=[0.0,0.999]
        zero_below_median (bool): set pixels below the median pixel value equal to zero before rescaling intensity, default=False
        verbose (bool): print info while processing or not, default=False
    Returns:
        out_array (np.array): 2D array, dtype=uint8
    """

    im = img_as_float64(im_array)
    q_low, q_high = np.quantile(im, clip_quantiles)
    im_clipped = np.clip(im, a_min=q_low, a_max=q_high)
    if zero_below_median:
        im_clipped[im_clipped < np.median(im_clipped)] = 0
    return img_as_ubyte_nowarn(rescale_intensity(im_clipped))


def _imagej_rewrite_autocontrast(
    im_array,
    upper_limit_frac=1 / 10,
    low_thresh_frac=1 / 5000,
    zero_high_count_pix=True,
    verbose=False,
    **kwargs
):
    r"""
    Imitation of imagej's autocontrast function
    Args:
        im_array (np.array): 2D array, usually dtype=uint16
        upper_limit_frac (float): the highest pixel value (in 8-bits) exceeding this fraction of all image pixels is used as the lower threshold for clipping the image, default=1/10
        low_thresh_frac (float): the lowest pixel value (in 8-bits) exceeding this fraction of all image pixels is used as the upper threshold for clipping the image, default=1/5000
        zero_high_count_pix (bool): set all pixel values below (inclusive) the upper_limit_frac are set to zero, default=True
        verbose (bool): print info while processing or not, default=False
    Returns:
        out_array (np.array): 2D array, dtype=uint8
    """

    # TODO don't convert to 8-bit and then stretch, stretch original image to get better sampling

    # convert to range 0,255 8-bit image if not already
    im_array_n = img_as_ubyte_nowarn(rescale_intensity(im_array))

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


CONTRAST_METHOD = {
    "imagej_rewrite": _imagej_rewrite_autocontrast,
    "simple_quantile": _simple_quantile_constrast,
}


def cell_worker(
    cell_label_value,
    Cautos=[],
    label_channel="cell",
    basename="unnamed_image_field",
    out_dir=None,
    channels=DEFAULT_CHANNELS,
    verbose=False,
):
    r"""
    segment single cells + save, log info to df
    Args:
        cell_label_value (int): integer mask value in segmentation for this cell,
        Cautos (list): auto-contrasted images from single iimage field, default=[],
        label_channel (str): name of channel to use as the image mask, default=="cell",
        basename (str): naem of the input image field, default=="unnamed_image_field",
        out_dir (str): where to save output images, default==None,
        channels (dict): {"name":index} map for input tiff, default=DEFAULT_CHANNELS
        verbose (bool): print info while processing or not, default=False
    Returns:
        cell_info_df (pd.DataFrame): info for each cell
    """
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)

    img_out_dir = out_dir.joinpath("output_single_cell_images")
    img_out_dir.mkdir(exist_ok=True)

    label_image = Cautos[channels[label_channel]]
    y, x = np.where(label_image == cell_label_value)
    crop_slice = np.s_[min(y) : max(y) + 1, min(x) : max(x) + 1]
    label_crop = label_image[crop_slice]
    mask = (label_crop == cell_label_value).astype(np.float64)

    cell_info_df = pd.DataFrame.from_records(
        list(channels.items()), columns=["channel_name", "channel_index"]
    )
    cell_info_df["field_image_name"] = basename
    cell_info_df["cell_label_value"] = cell_label_value
    cell_info_df["single_cell_channel_output_path"] = ""

    for i, row in cell_info_df.iterrows():
        channel, c = row["channel_name"], row["channel_index"]
        cell_object_crop = Cautos[c][crop_slice]
        cell_object_crop = cell_object_crop * mask
        cell_object_crop = img_as_ubyte_nowarn(rescale_intensity(cell_object_crop))

        out_filename = "{0}_cell{1}_C{2}.png".format(
            basename, cell_label_value, channel
        )
        out_path = img_out_dir.joinpath(out_filename)
        imageio.imwrite(out_path, cell_object_crop)
        cell_info_df.at[i, "single_cell_channel_output_path"] = out_path

    if verbose:
        print("cell_label_value = {}".format(cell_label_value))

    return cell_info_df


def read_and_contrast_image(
    image_path,
    image_dims="CYX",
    contrast_method="simple_quantile",
    contrast_kwargs=DEFAULT_CONTRAST_KWARGS,
    channels=DEFAULT_CHANNELS,
    channel_groups=DEFAULT_CHANNEL_GROUPS,
    verbose=False,
):
    r"""
    Load an image from a file path, return two lists: max projects per channel,
    and autocontrast versions of same.
    Args:
        image_path (str): location of input tiff image
        image_dims (str): input image dimension ordering, default="CYX"
        contrast_method (str): method for autocontrasting, default=="simple_quantile"
        contrast_kwargs (dict):, default=DEFAULT_CONTRAST_KWARGS
        channels (dict): {"name":index} map for input tiff, default=DEFAULT_CHANNELS
        channel_groups (dict): fluor/bf/seg grouping, default=DEFAULT_CHANNEL_GROUPS
        verbose (bool): print info while processing or not, default=False
    Returns:
        (Cmaxs, Cautos): tuple of two lists
            - unadjusted maxprojects per channel
            - stretched versions of same (for channels to be stretched, else unadjusted)
    """

    channel_types = {c: g for g, ms in channel_groups.items() for c in ms}

    # set which contrast function we're using
    contrast_fn = CONTRAST_METHOD[contrast_method]

    # set which contrast method gets applied to images vs segmentations
    contrast_fns = {
        grp: contrast_fn if grp != "seg" else img_as_ubyte_nowarn
        for grp, kwds in contrast_kwargs.items()
    }

    # read in all data for image and check that channel dim is correct length and all labeled
    im = AICSImage(image_path, known_dims=image_dims)
    assert dict(zip(im.dims, im.data.shape))["C"] == len(channels)

    # list of input max projects for each channels
    Cmaxs = [im.get_image_data("YX", C=c) for c in sorted(channels.values())]

    # auto contrast each channel according to what type of image it is
    Cautos = [
        contrast_fns[channel_types[c_name]](
            Cmaxs[c_ind], **contrast_kwargs[channel_types[c_name]]
        )
        for (c_name, c_ind) in channels.items()
    ]

    return Cmaxs, Cautos


def field_worker(
    image_path,
    image_dims="CYX",
    out_dir=None,
    contrast_method="simple_quantile",
    contrast_kwargs=DEFAULT_CONTRAST_KWARGS,
    channels=DEFAULT_CHANNELS,
    channel_groups=DEFAULT_CHANNEL_GROUPS,
    verbose=False,
):
    r"""
    Process an entire field -- autocontrast + save, log info to df
    Args:
        image_path (str): location of input tiff image
        image_dims (str): input image dimension ordering, default="CYX"
        out_dir (str): where to save output images, default=None
        contrast_method (str): method for autocontrasting, default=="simple_quantile"
        contrast_kwargs (dict):, default=DEFAULT_CONTRAST_KWARGS
        channels (dict): {"name":index} map for input tiff, default=DEFAULT_CHANNELS
        channel_groups (dict): fluor/bf/seg grouping, default=DEFAULT_CHANNEL_GROUPS
        verbose (bool): print info while processing or not, default=False
    Returns:
        field_info_df (pd.DataFrame): info for each field, merged with info for each cell
    """

    # early exit if file does not exist -- return df of just
    image_path = Path(image_path)
    if not image_path.is_file():
        return pd.DataFrame({"field_image_path": [image_path]})

    # set out dir if needed
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)

    # set out dir for rescaled images and create if needed
    output_field_image_dir = out_dir.joinpath("output_field_images")
    output_field_image_dir.mkdir(exist_ok=True)

    # contrast stretch the field channels
    Cmaxs, Cautos = read_and_contrast_image(
        image_path,
        contrast_method=contrast_method,
        contrast_kwargs=contrast_kwargs,
        channel_groups=channel_groups,
        image_dims=image_dims,
        verbose=verbose,
    )

    # save original and rescaled field data
    field_info_df = pd.DataFrame.from_records(
        list(channels.items()), columns=["channel_name", "channel_index"]
    )

    # save field level metadate to df
    field_info_df["field_image_path"] = ""
    field_info_df["rescaled_field_image_path"] = ""
    for i, row in field_info_df.iterrows():
        field_info_df.at[i, "field_image_path"] = image_path
        rescaled_field_channel_out_path = output_field_image_dir.joinpath(
            "{0}_C{1}.png".format(image_path.stem, row["channel_name"])
        )
        field_info_df.at[
            i, "rescaled_field_image_path"
        ] = rescaled_field_channel_out_path
        imageio.imwrite(rescaled_field_channel_out_path, Cautos[row["channel_index"]])

    # extract napari annotation channel and grab unique labels for cells
    label_image = Cautos[channels["cell"]]
    labels = np.unique(label_image)
    cell_labels = np.sort(labels[labels > 0])
    assert (cell_labels[0], cell_labels[-1]) == (1, len(cell_labels))

    if verbose:
        print("processing {}".format(image_path))
        print("found {} segmented cells".format(len(cell_labels)))

    # partial function for iterating over all cells in an image with map
    _cell_worker_partial = partial(
        cell_worker,
        Cautos=Cautos,
        label_channel="cell",
        basename=image_path.stem,
        out_dir=out_dir,
        channels=channels,
        verbose=verbose,
    )

    # iterate through all cells in an image
    all_cell_info_df = pd.concat(
        map(_cell_worker_partial, cell_labels), axis="rows", ignore_index=True
    )

    # merge cell-wise data into field-wise data
    all_cell_info_df["field_image_path"] = image_path
    field_info_df = field_info_df.merge(all_cell_info_df, how="inner")

    return field_info_df
