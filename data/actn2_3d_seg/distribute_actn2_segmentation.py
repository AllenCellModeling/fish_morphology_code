import os
from pathlib import Path
import pandas as pd
import quilt3
from quilt3distribute import Dataset

# Output folder where actn2 segmentation images are stored at
path = Path('/allen/aics/microscopy/Calysta/test/fish_struc_seg/output')

# Create data frame from the images
df = pd.DataFrame()
index = 0
for seg in os.listdir(path):
    if seg.endswith('.tiff'):
        index += 1
        row = {'index': str(index),
               'original_fov_name': seg.split('_C5_struct_segmentation.tiff')[0],
               'struc_seg_name': seg,
               'struc_seg_path': path + '/' + seg
               }
        df = df.append(row, ignore_index=True)

# Add original_fov_location in data frame
p_feats = quilt3.Package.browse(
        "tanyasg/2d_autocontrasted_single_cell_features",
        "s3://allencell-internal-quilt",
    )
df_feat_inds = p_feats["features"]["a749d0e2_cp_features.csv"]()[["fov_path"]].rename(columns={"fov_path":"original_fov_location"})
df_feat_inds = df_feat_inds.drop_duplicates()

for index, row in df_feat_inds.iterrows():
    df_feat_inds.loc[index, 'original_fov_name'] = row['original_fov_location'].split('/')[-1]

for index, row in df.iterrows():
    df.loc[index, 'original_fov_location'] = df_feat_inds.loc[df_feat_inds['file_name'] == row['original_fov_name'], 'original_fov_location'].values.tolist()[0]

# merge df
df_new = df.merge(df_feat_inds, how='inner', on=['original_fov_name'])
df_new = df_new.set_index('index')

# Upload to quilt
test_df = df_new[0:2]
ds = Dataset(
    dataset=df_new,
    name='3d_actn2_segmentation',
    package_owner='calystay',
    readme_path=r'C:\Users\calystay\Desktop\README.md',
)
ds.set_metadata_columns(["original_fov_location"])
ds.set_path_columns(["struc_seg_path"])
ds.distribute(
    "s3://allencell-internal-quilt", message="3D actn2 segmentation with original_fov_location"
    )

