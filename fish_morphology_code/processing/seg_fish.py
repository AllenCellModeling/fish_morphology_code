import numpy as np
from skimage.morphology import remove_small_objects

from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)
from aicssegmentation.core.output_utils import template_output, save_segmentation
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

from aicsimageprocessing import resize


def FABP3_Cardio_Pipeline(
    struct_img, rescale_ratio, output_type, output_path, fn, output_func=None
):
    ##########################################################################
    # PARAMETERS:
    #   note that these parameters are supposed to be fixed for the structure
    #   and work well accross different datasets

    # ADD you parameters here
    intensity_norm_param = [1, 18]
    minArea = 0
    gaussian_smoothing_sigma = 1
    # lower_bound = 685
    # upper_b = 18
    # ADD-HERE
    ##########################################################################

    out_img_list = []
    out_name_list = []

    ###################
    # PRE_PROCESSING
    ###################
    # intenisty normalization (min/max)
    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    # upper_bound = np.mean(struct_img) + 18 * np.std(struct_img)
    # struct_img[struct_img < lower_bound] = lower_bound
    # struct_img[struct_img > upper_bound] = upper_bound
    # struct_img = (struct_img - lower_bound).astype(float)/(upper_bound - lower_bound)

    out_img_list.append(struct_img.copy())
    out_name_list.append("im_norm")

    # rescale if needed
    if rescale_ratio > 0:
        struct_img = resize(
            struct_img, [1, rescale_ratio, rescale_ratio], method="cubic"
        )
        struct_img = (struct_img - struct_img.min() + 1e-8) / (
            struct_img.max() - struct_img.min() + 1e-8
        )

    # smoothing with gaussian filter
    structure_img_smooth = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma
    )  # ADD-HERE

    out_img_list.append(structure_img_smooth.copy())
    out_name_list.append("im_smooth")

    # core algorithm
    # th_li = threshold_li(structure_img_smooth)

    # PARAMETERS for this step ##
    s2_param = [[1, 0.02]]

    bw = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)

    # POST-PROCESSING

    seg = remove_small_objects(bw, min_size=minArea, connectivity=1, in_place=False)

    # output
    seg = seg > 0
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    out_img_list.append(seg.copy())
    out_name_list.append("bw_final")

    if output_type == "default":
        # the default final output
        save_segmentation(seg, False, output_path, fn)
    elif output_type == "TEMPLATE":
        # the hook for pre-defined output functions
        template_output(out_img_list, out_name_list, output_type, output_path, fn)
