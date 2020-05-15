import os
import torch
import numpy as np
import pandas as pd
from quilt3 import Package
from skimage import io as skio
from skimage.transform import resize
from skimage.morphology import label, skeletonize
from skimage.measure import block_reduce
from scipy.ndimage.morphology import distance_transform_edt as edt
from aicssegmentation.core.vessel import vesselness3D
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d

from cardio_CNN_classifier import cardio_cnn

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Download the model weights from Quilt if the folder `best_model` is empty

model_weights_path = []
if os.path.exists("../best_model"):
    for f in os.listdir("../best_model"):
        # Search for pth files
        if f.endswith(".pth"):
            model_weights_path.append(os.path.join("..", "best_model", f))
if not model_weights_path:
    # Download from Quilt
    print("No weights were found locally. Downloading from Quilt...")
    pkg = Package.browse(
        "matheus/assay_dev_actn2_classifier", "s3://allencell-internal-quilt"
    ).fetch("../best_model")
    metadata = pd.read_csv(os.path.join("..", "best_model", "metadata.csv"))
    model_weights_path = os.path.join("..", "best_model", metadata.model_path[0])
elif len(model_weights_path) > 1:
    # Use the last one in case more than 1 are found
    model_weights_path = model_weights_path[-1]
    print(f"More than 1 weight file found. Using the last one: {model_weights_path}.")
else:
    # Only one file found
    model_weights_path = model_weights_path[0]

# Load weights
classifier = cardio_cnn(model_path=model_weights_path)

# Segment the images for background calculation
def segment_image(struct_img):

    VESSELNESS_SIGMA = 1.0
    VESSELNESS_THRESHOLD = 1e-3

    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    response = vesselness3D(
        structure_img_smooth, sigmas=[VESSELNESS_SIGMA], tau=1, whiteonblack=True
    )

    return (response > VESSELNESS_THRESHOLD).astype(np.uint8)


# Downlaod the datasets from Quilt if there is no local copy
ds_folder = "../../database/"

if not os.path.exists(os.path.join(ds_folder, "metadata.csv")):

    pkg = Package.browse(
        "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
    ).fetch(ds_folder)

metadata = pd.read_csv(os.path.join(ds_folder, "metadata.csv"))

df_fov = pd.read_csv(os.path.join(ds_folder, metadata.database_path[0]), index_col=0)

df_cell = pd.read_csv(os.path.join(ds_folder, metadata.cell_database_path[0]))

# Run all FOVs
for index in df_fov.index:

    filename = df_fov.RawFileName[index]

    print("[{:03d}] - {:s}".format(index, filename))

    # Load data
    nuc_ch = 1  # Channel of nuclar dye
    str_ch = 5  # Channel of EGFP-alpha-actinin-2

    img = skio.imread(filename)
    print(img.shape)
    nuc = img[int(nuc_ch)]
    img = img[int(str_ch)]

    # Load single cell segmentation
    df_cell_sub = df_cell.loc[df_cell.RawFileName == df_fov.RawFileName[index]]
    mask_name = df_cell_sub.MaskFileName.values[0]
    single_cell_mask = skio.imread(mask_name)[-1]

    # Structure segmentation
    nslices = img.shape[0]
    zprofile = img.mean(axis=-1).mean(axis=-1)
    slice_number = np.argmax(zprofile)

    img = img[slice_number - 3 : slice_number + 4]
    img_binary = segment_image(img)

    data = img[3]
    data_binary = img_binary[3]

    # Detecting background
    data_cc = data_binary.copy()
    data_cc = data_cc.astype(np.uint32)
    data_cc = label(data_binary)

    data_skell = data_binary.copy()
    data_skell = skeletonize(data_skell)
    data_skell = data_skell.astype(np.uint8)

    data_junc = data_skell.copy()
    for d in [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
        data_junc[2:-2, 2:-2] += data_skell[2 + d[0] : -2 + d[0], 2 + d[1] : -2 + d[1]]

    data_lines = data_skell.copy()
    data_lines[data_junc > 3] = 0

    bkrad = 64
    data_mask = data_binary.copy()
    data_mask = data_mask.astype(np.uint16)
    data_tmp = block_reduce(data_mask, (bkrad, bkrad), np.sum)
    data_mask = (
        resize(
            data_tmp,
            data_mask.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
            mode="reflect",
        )
        < bkrad
    ).astype(np.uint8)
    data_label_fine = label(data_mask)
    data_label_coarse = label(edt((data_mask > 0)) > bkrad)
    for lb_c in np.unique(data_label_coarse.reshape(-1)):
        if lb_c > 0:
            y, x = np.where(data_label_coarse == lb_c)
            lb_f = np.unique(data_label_fine[y, x])
            data_label_fine[data_label_fine == lb_f] = -1
    data_mask = (data_label_fine < 0).astype(np.uint8)

    # CNN inference
    data_probs = classifier.predict_image_sliding(
        raw_im=data, stride=8, batch_size=120, interp_order=1, device=device
    )

    # Reordering the classes to
    # 1 - Diffuse/others
    # 2 - Fibers
    # 3 - Disorganized puncta
    # 4 - Organized Puncta
    # 5 - Organized Z-disks
    data_probs_bkp = data_probs.copy()
    for ulabel in [(1, 5), (2, 1), (3, 4), (4, 2), (5, 3)]:
        data_probs[ulabel[1] - 1] = data_probs_bkp[ulabel[0] - 1]
    data_classification = np.argmax(data_probs, axis=0) + 1
    data_classification = data_classification.astype(np.uint8)

    # Masking background out
    data_probs[:, data_mask > 0] = 0
    data_classification[data_mask > 0] = 0

    # Save data into single TIFF file
    dim = data.shape
    data_output = np.vstack(
        [
            data.reshape(-1, *dim),
            data_probs,
            data_classification.reshape(-1, *dim),
            single_cell_mask.reshape(-1, *dim),
        ]
    ).astype(np.float32)

    if not os.path.exists("../../output/"):
        os.makedirs("../../output/")

    skio.imsave("../../output/fov_{0}.tif".format(index), data_output)
