import os
import pandas as pd

output_folder = r'\\allen\aics\microscopy\Calysta\test\fish_struc_seg'
output_df = pd.read_csv(os.path.join(output_folder, 'sarc_classification.csv'))
# attach probe id to each cell
cp_df = pd.read_csv(r'\\allen\aics\microscopy\Calysta\test\fish_struc_seg\radon_update\cp_df_2.csv')

output_df['cell_num'] = output_df['cell_num'].astype(int)
for seg in ['561', '638']:
    for class_type in range (0, 6):

        # add density inside class
        output_df['seg_' + seg + '_density_class_' + str(class_type)] = output_df['seg_' + seg + '_probe_px_class_' + str(class_type)] / output_df['seg_' + seg + '_area_px_class_' + str(class_type)]

        # calculate density outside of class
        probe_px_out = 0
        area_px_out = 0
        for out_class_type in range (0, 6):
            if out_class_type != class_type:
                probe_px_out += output_df['seg_' + seg + '_probe_px_class_' + str(out_class_type)]
                area_px_out += output_df['seg_' + seg + '_area_px_class_' + str(out_class_type)]
        # add density outside of class
        output_df['seg_' + seg + '_probe_px_OUTSIDE_class_' + str(class_type)] = probe_px_out
        output_df['seg_' + seg + '_area_px_OUTSIDE_class_' + str(class_type)] = area_px_out
        output_df['seg_' + seg + '_density_OUTSIDE_class_' + str(class_type)] = probe_px_out/area_px_out

# Turn nan value to 0, denominator = 0
for index, row in output_df.iterrows():
    for seg in ['561', '638']:
        for class_type in range (0, 6):
            if row['seg_' + seg + '_area_px_class_' + str(class_type)] == 0:
                output_df.loc[index, 'seg_' + seg + '_density_class_' + str(class_type)] = 0

# Add column of fov path to match with other csvs
for index, row in output_df.iterrows():
    fov_path = row['RawPath'].split('_C0')[0]
    output_df.loc[index, 'fov_path'] = fov_path

# Clean columns
for class_type in range (0, 6):
    output_df['area_px_class_' + str(class_type)] = output_df['seg_561_area_px_class_' + str(class_type)]
    for seg in ['561', '638']:
        output_df.drop(columns=['seg_' + seg + '_area_px_class_' + str(class_type)])

# Output to csv
output_df.to_csv(r'\\allen\aics\microscopy\Calysta\test\fish_struc_seg\sarc_classification_for_Rory_20200210.csv')