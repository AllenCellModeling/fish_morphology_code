import os
import numpy as np
import pandas as pd
from skimage import io as skio
from skimage.transform import resize
from skimage.filters import median
from skimage.morphology import label, skeletonize
from skimage.measure import block_reduce
from skimage.morphology import binary_dilation, square
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage import segmentation as skseg
from scipy.ndimage import morphology as scimorpho

from aicssegmentation.core.vessel import vesselness3D
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d

from cardio_CNN_classifier import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
classifier = cardioCNN(model_path='cnn_model/res18_final_e708.pth')

#
# Aux
#

def SegmentImage(struct_img):
    
    INTENSITY_NORM_PARAM = [5, 15]
    VESSELNESS_SIGMA = 1.0
    VESSELNESS_THRESHOLD = 1e-3

    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)  

    response = vesselness3D(structure_img_smooth, sigmas=[VESSELNESS_SIGMA],  tau=1, whiteonblack=True)
   
    return (response>VESSELNESS_THRESHOLD).astype(np.uint8)

#
# Parameters
#

database = pd.read_csv('database/database.csv', index_col=0)

for index in database.index:

    filename = database.RawFileName[index]

    print("[{:03d}] - {:s}".format(index,filename))

    #
    # Load data
    #

    nuc_ch = 1
    str_ch = 5

    img = skio.imread(filename)
    print(img.shape)
    nuc = img[int(nuc_ch)]
    img = img[int(str_ch)]


    #
    # Structure segmentation
    #

    nslices = img.shape[0]
    zprofile = img.mean(axis=-1).mean(axis=-1)
    slice_number = np.argmax(zprofile)

    img = img[slice_number-3:slice_number+4]
    img_binary = SegmentImage(img)

    data = img[3]
    data_binary = img_binary[3]

    #
    # Nuclei segmentation
    #

    nuc = nuc[slice_number]
    nuc = median(nuc, np.ones((20,20)))
    data_nuc_binary = median(nuc > np.percentile(nuc, 95), np.ones((20,20)))

    #
    # Detecting background
    #

    data_cc = data_binary.copy()
    data_cc = data_cc.astype(np.uint32)
    data_cc = label(data_binary)

    data_skell = data_binary.copy()
    data_skell = skeletonize(data_skell)
    data_skell = data_skell.astype(np.uint8)

    data_junc = data_skell.copy()
    for d in [(0,-1),(1,0),(0,1),(-1,0),(-1,-1),(1,-1),(1,1),(-1,1)]:
        data_junc[2:-2,2:-2] += data_skell[2+d[0]:-2+d[0],2+d[1]:-2+d[1]]

    data_lines = data_skell.copy()
    data_lines[data_junc>3] = 0

    bkrad = 64
    data_mask = data_binary.copy()
    data_mask = data_mask.astype(np.uint16)
    data_tmp = block_reduce(data_mask, (bkrad,bkrad), np.sum)
    data_mask = (resize(data_tmp, data_mask.shape, order=0, preserve_range=True, anti_aliasing=False, mode='reflect')<bkrad).astype(np.uint8)
    data_label_fine = label(data_mask)
    data_label_coarse = label(edt((data_mask>0))>bkrad)
    for lb_c in np.unique(data_label_coarse.reshape(-1)):
        if lb_c > 0:
            y, x = np.where(data_label_coarse==lb_c)
            lb_f = np.unique(data_label_fine[y,x])
            data_label_fine[data_label_fine==lb_f] = -1
    data_mask = (data_label_fine<0).astype(np.uint8)
    #data_mask[data_nuc_binary>0] = 1

    #
    # Calculating Voronoi
    #

    data_lines_cc = label(data_lines)
    if data_lines_cc.max() < 65535:
        data_lines_cc = data_lines_cc.astype(np.uint16)

    data_lines_edtidxs = scimorpho.distance_transform_edt(1-data_lines, return_distances=False, return_indices=True)

    coordy = data_lines_edtidxs[0].flatten()
    coordx = data_lines_edtidxs[1].flatten()

    data_voronoi = data_lines_cc[coordy,coordx].reshape(data_lines_cc.shape).copy()

    for u in range(1, 1+data_voronoi.max()):
        if np.sum(data_mask[data_voronoi==u], axis=None) > 0:
            data_voronoi[data_voronoi==u] = 0

    #
    # CNN inference
    #

    data_probs = classifier.predict_image_sliding(raw_im=data, stride=8, batch_size=120, interp_order=1, device=device)

    data_probs_bkp = data_probs.copy() #order labels as messy, threads, random dots, regular dots and regular stripes
    for ulabel in [(1,5),(2,1),(3,4),(4,2),(5,3)]:
        data_probs[ulabel[1]-1] = data_probs_bkp[ulabel[0]-1]
    data_classification = (np.argmax(data_probs, axis=0) + 1)
    data_classification = data_classification.astype(np.uint8)

    #
    # Save data into single TIFF file
    #

    dim = data.shape
    data_output = np.vstack([
        data.reshape(-1,*dim),
        data_binary.reshape(-1,*dim),
        data_mask.reshape(-1,*dim),
        data_voronoi.reshape(-1,*dim),
        data_probs,
        data_classification.reshape(-1,*dim)]).astype(np.float32)

    skio.imsave('data_output_fov_{0}.tif'.format(index), data_output)
