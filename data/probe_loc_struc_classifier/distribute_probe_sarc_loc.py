import pandas as pd
from quilt3distribute import Dataset

df = pd.read_csv('/allen/aics/microscopy/Calysta/test/fish_struc_seg/sarc_classification_for_Rory.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['Unnamed: 0.1'], axis=1)

df = df.rename(columns = {'fov_path': 'original_fov_location',
                          'cell_num': 'napariCell_ObjectNumber'})

df = df[['nuc_mask_path', 'original_fov_location']]
df = df.drop_duplicates()
# df = df.drop(['RawPath'], axis=1)

test_df = df.loc[0:2]
ds = Dataset(
    dataset=test_df,
    name='2d_nuclear_masks_test',
    package_owner='calystay',
    readme_path=r'C:\Users\calystay\Desktop\README.md',
)
ds.set_metadata_columns(["original_fov_location"])
ds.set_path_columns(["nuc_mask_path"])
ds.distribute(
    "s3://allencell-internal-quilt", message="2D nuclear masks with original_fov_location"
    )

