import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io as skio
from skimage import segmentation as skseg
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def ProcessFOV(FOVId):

	try:

		data = skio.imread('/allen/aics/assay-dev/MicroscopyOtherData/Viana/CardioOrganization/data_output_fov_{0}_radon_w64.tif'.format(FOVId))

	except: return None

	cla = data[9].astype(np.uint8)
	raw = data[0].astype(np.float32)
	msk = data[-1].astype(np.uint8)

	for CellId in np.unique(msk):

		if CellId:

			cell_raw = raw.copy()

			[j,i] = np.where(msk!=CellId)
			cell_raw[j,i] = 0
			[j,i] = np.where(msk==CellId)
			cell_raw = cell_raw[j.min():j.max(),i.min():i.max()]
			cell_cla = cla[j.min():j.max(),i.min():i.max()]

			h, w = cell_raw.shape

			cell_rgb = np.zeros((h,w,3), dtype = np.uint8)

			qinf, qsup = np.percentile(cell_raw[cell_raw>0].flatten(),[5,95])

			cell_raw = np.clip(cell_raw,qinf,qsup)
			cell_raw = (cell_raw-qinf)/(qsup-qinf)

			for label, r, g, b in [[5,250,230,30],[1,70,0,80],[4,110,205,90],[2,65,65,135],[3,35,135,140]]:

				cell_raw_label = cell_raw[cell_cla==label]

				cell_rgb[cell_cla==label,0] = cell_raw_label * r
				cell_rgb[cell_cla==label,1] = cell_raw_label * g
				cell_rgb[cell_cla==label,2] = cell_raw_label * b

			con = skseg.find_boundaries(msk[j.min():j.max(),i.min():i.max()]==CellId)
			[j,i] = np.nonzero(con)
			cell_rgb[j,i,:] = 255

			skio.imsave('/allen/aics/assay-dev/MicroscopyOtherData/Viana/CardioOrganization/thumbs/fov-{0}-cell-{1}.jpg'.format(FOVId,CellId), cell_rgb)

	return 1

if __name__ == "__main__":

    #
    # Get FOV id
    #

	database = pd.read_csv('database/database.csv', index_col=0)

	Parallel(n_jobs=20)(delayed(ProcessFOV)(FOVId) for FOVId in tqdm(database.index))

    
