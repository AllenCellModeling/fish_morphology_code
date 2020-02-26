import pandas as pd
from quilt3distribute import Dataset

df = pd.read_csv('/allen/aics/microscopy/Calysta/test/fish_struc_seg/dist_radon_transform_all.csv')

# Add original_fov_location in dist_radon_transform_output

# Convert pipeline name to original_fov_location
home_directory = '/allen/aics/assay-dev/MicroscopyData/Melissa/2019'
end_string = '_C0.tif'


def convert_img_name_to_original_fov_location(img_name, end_string, home_directory):
    plate, obj_mag, date, session, position, well = img_name.split('_')
    original_name = date + '_M01_00' + session[1:] + '_Capture 1 - Capture 1 - ' + well + ' [144] Montage - ' + position[1:] + end_string
    original_fov_store = home_directory + '/' + date + '/' + obj_mag[0:2] + 'x_exports'
    original_fov_location = original_fov_store + '/' + original_name
    return original_fov_location



for index, row in df.iterrows():
    image_name = row['image_name']
    df.loc[index, 'original_fov_location'] = convert_img_name_to_original_fov_location(image_name, end_string, home_directory)

test_df = df.loc[0:2]
ds = Dataset(
    dataset=df,
    name='probe_localization',
    package_owner='calystay',
    readme_path=r'C:/Users/calystay/Desktop/README.md',
)
ds.set_metadata_columns(["original_fov_location"])
ds.distribute(
    "s3://allencell-internal-quilt", message="probe localization with original_fov_location"
    )

