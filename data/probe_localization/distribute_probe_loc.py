import pandas as pd
import quilt3
from quilt3distribute import Dataset
from fish_morphology_code.analysis.collate_plot_dataset import collate_plot_dataset


df = pd.read_csv('/allen/aics/microscopy/Calysta/test/fish_struc_seg/dist_radon_transform_all.csv')

# Add original_fov_location in dist_radon_transform_output

p_feats = quilt3.Package.browse(
        "tanyasg/2d_autocontrasted_single_cell_features",
        "s3://allencell-internal-quilt",
    )
df_feat_inds = p_feats["features"]["a749d0e2_cp_features.csv"]()[["fov_path", "napariCell_ObjectNumber"]].rename(columns={"fov_path":"original_fov_location"})
df_feat_inds.drop_duplicates()

plates_date = {'20190807': '5500000013',
               '20190816': '5500000014'}

plot_ds = collate_plot_dataset()

def convert_original_fov_location_to_img_name(original_fov_location, plates_date):
    file_name = original_fov_location.split('/')[-1]
    session = file_name.split('_')[2][2]
    well = file_name.split(' [144]')[0][-2:]
    position_endstring = file_name.split('-')[-1][1:]
    position = position_endstring.split(file_name)[0]

    date_of_file = file_name.split('_')[0]
    plate = str(plates_date[date_of_file])

    data_file = str(plate) + '_63X_' + date_of_file + '_S' + session + '_P' + position + '_' + well
    return data_file


for index, row in df_feat_inds.iterrows():
    image_name = convert_original_fov_location_to_img_name(row["original_fov_location"], plates_date)
    df_feat_inds.loc[index, 'image_name'] = image_name

df = df.rename(columns={"object_number": "napariCell_ObjectNumber"})

for index, row in df.iterrows():
    image_name = row['image_name']
    location = list(set(df_feat_inds.loc[df_feat_inds['image_name'] == image_name, 'original_fov_location']))[0]
    df.loc[index, 'original_fov_location'] = location

plot_df = plot_ds.merge(right=df,
                        left_on=['FOV path', 'Cell number'],
                        right_on=['original_fov_location', 'napariCell_ObjectNumber'])

plot_df = plot_df[['original_fov_location', 'napariCell_ObjectNumber',
                   'seg_561_cell_dist_nuc_per_obj_median',
                   'seg_638_cell_dist_nuc_per_obj_median']]

plot_df.to_csv('probe_localization_for_plot.csv')

test_df = df.loc[0:2]
ds = Dataset(
    dataset=df,
    name='probe_localization',
    package_owner='calystay',
    readme_path='C:/Users/calystay/Desktop/README.md',
)
ds.set_extra_files(['probe_localization_for_plot.csv'])
ds.set_metadata_columns(["original_fov_location"])
ds.distribute(
    "s3://allencell-internal-quilt", message="probe localization with original_fov_location"
    )
