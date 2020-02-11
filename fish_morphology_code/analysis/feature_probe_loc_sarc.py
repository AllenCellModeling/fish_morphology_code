import numpy as np
import os
import pandas as pd
import tifffile

seg_folder = '/allen/aics/modeling/data/cardio_fish/normalized_2D_tiffs/output_field_images'
class_folder = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/output'
nuc_folder = '/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20191122/cp_out_images'
output_folder = '/allen/aics/microscopy/Calysta/test/fish_struc_seg'
nuc_csv = pd.read_csv('/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20191122/absolute_metadata.csv')
nuc_csv = nuc_csv[['original_fov_location', 'rescaled_2D_fov_tiff_path']].drop_duplicates()

sarc_csv = pd.read_csv('/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/AssayDevFishAnalsysis2019-Handoff.csv')

for index, row in sarc_csv.iterrows():
    sarc_csv.loc[index, 'fovid'] = row['CellId'].split('-')[1]
sarc_names = sarc_csv[['ImagePath', 'RawPath', 'fovid']]
sarc_names = sarc_names.drop_duplicates()

plates_date = {'20190807': '5500000013',
               '20190816': '5500000014'}
end_string = '_C0.tif'
channel_dict = {'bf':0, '488':1, '561':2, '405':3, '638':4,
                'seg_probe_488':5, 'seg_probe_561':6, 'seg_probe_638':7,
                'foreback':8, 'cell':9}
probe_segs = {6:'seg_561', 7:'seg_638'}

output_df = pd.DataFrame()
count = 0
for index, row in sarc_names.iterrows():
    img_path = row['RawPath']
    img_name = img_path.split('/')[-1]
    print ('reading ' + str(count) + ' ' + img_name)
    count += 1
    session = img_name.split('_')[2][2]
    well = img_name.split(' [144]')[0][-2:]
    position_endstring = img_name.split('-')[-1][1:]
    position = position_endstring.split(end_string)[0]

    date_of_file = img_name.split('_')[0]
    plate = str(plates_date[date_of_file])

    data_file = str(plate) + '_63X_' + date_of_file + '_S' + session + '_P' + position + '_' + well
    seg_img_path = os.path.join(seg_folder, data_file + '_annotations_corrected_rescaled.ome.tiff')
    class_img_path = os.path.join(class_folder, row['ImagePath'].split('/')[-1].split('radon')[0] + 'bkgrd.tif')

    seg_img = tifffile.imread(seg_img_path)
    class_img = tifffile.imread(class_img_path)

    napari = seg_img[channel_dict['cell'], :, :]

    img_path_tif = img_path.split(end_string)[0]
    nuc_name = nuc_csv.loc[nuc_csv['original_fov_location'] == img_path_tif, 'rescaled_2D_fov_tiff_path'].tolist()[0].split('/')[-1].split('.tiff')[0] + 'nuc_final_mask.tiff'
    nuc_mask = tifffile.imread(os.path.join(nuc_folder, nuc_name))

    nuc_mask_binary = np.zeros(nuc_mask.shape)
    nuc_mask_binary[nuc_mask == 0] = 1

    cell_masked_nuc = (napari * nuc_mask_binary)

    for cell in np.unique(cell_masked_nuc):
        if cell > 0:

            data = {'nuc_mask_path': os.path.join(nuc_folder, nuc_name),
                    'RawPath': img_path,
                    'cell_num': cell}

            cell_mask = cell_masked_nuc==cell
            class_mask = class_img[9, :, :] * cell_mask

            cell_px = np.sum(napari==cell)

            nuc = (napari == cell) * nuc_mask
            nuc = nuc.astype(bool)
            nuc_px = np.sum(nuc)
            data.update({'cell_px': cell_px,
                         'nuc_px': nuc_px})

            for probe_channel, seg_id in probe_segs.items():
                # probe_channel = 6 # TODO: delete this
                # seg_id = 'seg_561'
                probe_seg = seg_img[probe_channel, :, :]
                probe_in_mask = (cell_mask * probe_seg) > 0

                total_probe_px = np.sum(probe_in_mask)

                data.update({seg_id + '_total_probe_cyto': total_probe_px})

                for sarc_class in range (1, 7):
                    probe_px = np.sum(probe_in_mask * (class_mask==sarc_class))
                    class_px = np.sum(class_mask==sarc_class)
                    data.update({seg_id + '_probe_px_class_' + str(sarc_class): probe_px,
                                 seg_id + '_area_px_class_' + str(sarc_class): class_px})

                probe_in_nucleus = nuc * probe_seg
                probe_in_nucleus = probe_in_nucleus.astype(bool)
                probe_nuc_px = np.sum(probe_in_nucleus)

                data.update({seg_id + '_probe_px_nuc': probe_nuc_px})

            output_df = output_df.append(data, ignore_index=True)
            output_df.to_csv(os.path.join(output_folder, 'sarc_classification_temp.csv'))

output_df.to_csv(os.path.join(output_folder, 'sarc_classification.csv'))


