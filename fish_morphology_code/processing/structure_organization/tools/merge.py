import os
import numpy as np
import pandas as pd
from tqdm import tqdm

source = '../output/'

database = pd.read_csv('../database/database.csv', index_col=0)

df = []

for FOVId in tqdm(database.index):

	try:

		df_fov = pd.read_csv(os.path.join(source,f'fov_{FOVId}.csv'))
		df_fov['FOVId'] = FOVId
		df_fov['ImagePath'] = f'/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/output/fov_{FOVId}.tif'
		df_fov['RawFileName'] = database.RawFileName[FOVId]
		df_fov['original_fov_location'] = database.RawFileName[FOVId].replace('_C0.tif','')
		df.append(df_fov)

	except:
		
		print(f'Data for FOV {FOVId} not found')
		pass

df = pd.concat(df,axis=0).reset_index(drop=True)

df['NewCellId'] = None
for index in df.index:
	FOVId = df.FOVId[index]
	CellId = df.CellId[index]
	df.loc[index,'NewCellId'] = 'fov-{0}-cell-{1}'.format(FOVId,CellId)
	df.loc[index,'napariCell_ObjectNumber'] = CellId

df = df.drop(['FOVId','CellId'],axis=1).rename(columns={'NewCellId':'CellId'}).set_index(['CellId'])

print(df.head())

df['Structure'] = 'ACTN2'

df.to_csv(os.path.join(source,'FishAnalsysis2020.csv'))

print(df.columns.tolist())