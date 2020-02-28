import os
from pathlib import Path
import pandas as pd
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
               'original_fov_location': seg.split('_C5_struct_segmentation.tiff')[0],
               'struc_seg_name': seg,
               'struc_seg_path': path + '/' + seg
               }
        df = df.append(row, ignore_index=True)

df = df.set_index('index')

# Upload to quilt
test_df = df[0:2]
ds = Dataset(
    dataset=df,
    name='3d_actn2_segmentation',
    package_owner='calystay',
    readme_path=r'C:\Users\calystay\Desktop\README.md',
)
ds.set_metadata_columns(["original_fov_location"])
ds.set_path_columns(["struc_seg_path"])
ds.distribute(
    "s3://allencell-internal-quilt", message="3D actn2 segmentation with original_fov_location"
    )

