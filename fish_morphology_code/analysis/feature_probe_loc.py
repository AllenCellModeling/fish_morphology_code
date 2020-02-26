import numpy as np
import os
from pathlib import Path
from aicsimageio import AICSImage
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
from skimage import measure, filters
from skimage.measure import regionprops
import tifffile

def generate_per_cell_region (cell_mask, struc_mask, nuc_mask, img, probe_segs):
    masks = {'cell': cell_mask,
             'nuc': nuc_mask,
             'struc': struc_mask}

    # set bounding box
    for rgn, mask in masks.copy().items():
        mask_reverse = np.zeros(mask.shape)
        mask_reverse[mask==0] = 1
        masks[rgn + '_reverse'] = mask_reverse

    # Draw bounding box
    y_lower = np.min(np.where(masks['cell']==1)[0])
    y_higher = np.max(np.where(masks['cell']==1)[0])
    x_lower = np.min(np.where(masks['cell']==1)[1])
    x_higher = np.max(np.where(masks['cell']==1)[1])

    if y_lower-5 > 0:
        y_lower = y_lower-5
    if x_lower-5 > 0:
        x_lower = x_lower-5
    if y_higher+5 > 0:
        y_higher = y_higher+5
    if x_higher+5 > 0:
        x_higher = x_higher+5

    box_masks = {}
    for rgn, mask in masks.items():
        bound_box = mask[y_lower:y_higher, x_lower:x_higher]
        box_masks[rgn] = bound_box

    for channel, seg_id in probe_segs.items():
        mask = img[channel, :, :] * cell_mask
        boxed = mask[y_lower:y_higher, x_lower:x_higher]
        box_masks[seg_id] = boxed

    return masks, box_masks


def calculate_transform_in_cell (metric, array, bin, dist_norm, seg_id, region, mode):
    fea_metric = {

        seg_id + '_' + region + '_' + metric + '_' + mode + '_median': np.median(array[0 <= array]),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_mean': np.mean(array[0 <= array]),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_std': np.std(array[0 <= array]),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_sum': np.sum(array[0 <= array]),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_c25': np.nanpercentile(array[0 <= array], 25),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_c50': np.nanpercentile(array[0 <= array], 50),
        seg_id + '_' + region + '_' + metric + '_' + mode + '_c75': np.nanpercentile(array[0 <= array], 75)
    }
    if bin:
        fea_metric.update({
            seg_id + '_' + region + '_' + metric + '_' + mode + '_bin25count': ((0 <= array) & (array < 0.25)).sum(),
            seg_id + '_' + region + '_' + metric + '_' + mode + '_bin50count': ((0.25 <= array) & (array < 0.5)).sum(),
            seg_id + '_' + region + '_' + metric + '_' + mode + '_bin75count': ((0.5 <= array) & (array < 0.75)).sum(),
            seg_id + '_' + region + '_' + metric + '_' + mode + '_bin100count': (array >= 0.75).sum(),
        })
        if mode == 'per_obj':
            # Calculate per object normalization
            fea_metric.update({
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_area_bin25count': ((0<=array) & (array<0.25)).sum() / ((dist_norm>=0) & (dist_norm<0.25)).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_area_bin50count': ((0.25<=array) & (array<0.5)).sum() / ((dist_norm>=0.25) & (dist_norm<0.5)).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_area_bin75count': ((0.5<=array) & (array<0.75)).sum() / ((dist_norm>=0.5) & (dist_norm<0.75)).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_area_bin100count': (0.75 <= array).sum() / (dist_norm>=0.75).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_count_bin25count': ((0 <= array) & (array < 0.25)).sum() / (array>=0).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_count_bin50count': ((0.25 <= array) & (array < 0.5)).sum() / (array>=0).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_count_bin75count': ((0.5 <= array) & (array < 0.75)).sum() / (array>=0).sum(),
                seg_id + '_' + region + '_' + metric + '_' + mode + '_norm_count_bin100count': (0.75 <= array).sum() / (array>0).sum()
            })

    return fea_metric


def generate_coordinate_pair(obj):
    coordinates = np.where(obj >= 1)
    obj_coor = []
    for px in range(0, len(coordinates[0])):
        y = coordinates[0][px]
        x = coordinates[1][px]
        obj_coor.append((y, x))
    return obj_coor


def calculate_distance_with_obj_seed (obj_seed, obj_ref_reverse, set_coor):
    obj_ref_coor = generate_coordinate_pair(obj_ref_reverse)
    obj_seed_coor = generate_coordinate_pair(obj_seed)
    dist_values = []
    for coor_pair in set_coor:
        dist_ref = distance.cdist(obj_ref_coor, [coor_pair], 'euclidean')
        dist_seed = distance.cdist(obj_seed_coor, [coor_pair], 'euclidean')
        #print(coor_pair, np.min(dist_ref), np.min(dist_seed))
        ratio = np.min(dist_seed) / (np.min(dist_ref) + np.min(dist_seed))
        dist_values.append(ratio)
    return dist_values


# ======================================================================================================================
# Main function

plates_date = {'20190807': '5500000013',
               '20190816': '5500000014'}

all_data_folder = Path('/allen/aics/gene-editing/FISH/2019/chaos/data/merged_napari')
all_seg_folder = Path('/allen/aics/microscopy/Calysta/test/fish_struc_seg/output')

channel_dict = {
    'bf':0,
    '488':1,
    '561':2,
    '405':3,
    '638':4,
    'seg_probe_488':5,
    'seg_probe_561':6,
    'seg_probe_638':7,
    'foreback':8,
    'cell':9
}

probe_segs = {
    6:'seg_561',
    7:'seg_638'
}

end_string = '_C5_struct_segmentation.tif'

rt_dict = {
    0:'angle',
    1:'org'
}

df = pd.DataFrame()

df_output_folder = Path('/allen/aics/microscopy/Calysta/test/fish_struc_seg')
nuc_mask_folder = Path('/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20191015/cp_out_images')
radon_org_folder = Path('/allen/aics/microscopy/Calysta/test/fish_struc_seg/radon_t_20190916')

cp_csv = Path('some-output-csv-from-cell-profiler-processing')

cp_df = pd.read_csv(cp_csv)

for index, row in cp_df.iterrows():
    file_name = row['ImagePath'].split('/')[-1]
    cp_df.loc[index, 'file_name'] = file_name

failed_img = []
df_failed_cell = pd.DataFrame()
img_count = 1

seg_imgs = os.listdir(all_seg_folder)
for seg_file in seg_imgs:
    if seg_file.endswith('.tiff'):
        try:
            # Read segmentation image, convert to mip in xy
            print ('reading ' + str(img_count) + ' image, filename: ' + seg_file)
            seg_data = tifffile.imread(os.path.join(all_seg_folder, seg_file))
            seg = seg_data
            seg_xy = np.amax(seg, axis=0)

            # Map segmentation image with annotated image with all data
            session = seg_file.split('_')[2][2]
            well = seg_file.split(' [144]')[0][-2:]
            position_endstring = seg_file.split('-')[-1][1:]
            position = position_endstring.split(end_string)[0]

            date_of_file = seg_file.split('_')[0]
            plate = str(plates_date[date_of_file])

            data_file = str(plate) + '_63X_' + date_of_file + '_S' + session + '_P' + position + '_' + well
        except:
            print('cannot process image ' + seg_file)
            failed_img.append(seg_file)
            seg = None
            pass
        img_count += 1
        if seg is not None:
            # Read annotated image
            try:
                data_data = tifffile.imread(os.path.join(all_data_folder, plate, data_file + '_annotations_corrected.tiff'))
                napari = data_data[channel_dict['cell'], :, :]
                nuc_mask_data = tifffile.imread(os.path.join(nuc_mask_folder, data_file + '_annotations_corrected_rescaled.omenuc_final_mask.tiff'))
                nuc = nuc_mask_data
                radon_org_data = tifffile.imread(os.path.join(radon_org_folder, 'org_mask_' + data_file + '_annotations_corrected.tiff'))
                radon_org = radon_org_data
                radon_org = radon_org/np.max(radon_org)
            except:
                napari=None
                print('cannot process image ' + seg_file)
                failed_img.append(seg_file)
                pass

            if napari is not None:
                for cell_object in range (1, int(np.max(napari)) + 1):
                    row = {'image_name': data_file,
                           'mask_name': 'napari',
                           'object_number': cell_object}
                    print (row)
                    try:
                        cell_mask = napari == cell_object

                        nuc_num = int(cp_df.loc[((cp_df['file_name'] == (data_file+'_annotations_corrected_rescaled.ome.tiff')) & (cp_df['napariCell_ObjectNumber'] == float(cell_object))),
                                                'finalnuc_ObjectNumber'])


                        nuc_mask = nuc == nuc_num
                        label_nuc_mask = measure.label(nuc_mask)
                        props = regionprops(label_nuc_mask)
                        nuc_centroid = props[0].centroid

                        struc_mask = seg_xy * cell_mask

                        object_masks, boxed_masks = generate_per_cell_region(cell_mask=cell_mask, nuc_mask=nuc_mask,
                                                                             struc_mask=struc_mask, img=data_data,
                                                                             probe_segs=probe_segs)

                        cell_dist_map = ndimage.distance_transform_edt(cell_mask)
                        dist_norm = cell_dist_map/np.max(cell_dist_map)

                        radon_org_mask = radon_org*cell_mask

                        struc_rev_dt = ndimage.distance_transform_edt(boxed_masks['struc_reverse'])
                        nuc_rev_dt = ndimage.distance_transform_edt(boxed_masks['nuc_reverse'])

                        for probe_channel, seg_id in probe_segs.items():
                            probe_seg = data_data[probe_channel, :, :]
                            probe_in_mask = (cell_mask * probe_seg) > 0

                            # Calculate centroid locations for probes
                            probe_centers = []
                            label_probe = measure.label(probe_in_mask)
                            props = regionprops(label_probe)
                            for probe in range (0, len(props)):
                                y = int(props[probe].centroid[0])
                                x = int(props[probe].centroid[1])
                                probe_centers.append((y, x))

                            row.update({seg_id + '_probe_center_loc': probe_centers})

                            # Map probe on distance transform on cell per probe_area
                            dist_mask = dist_norm * probe_in_mask
                            dist_mask[dist_mask == 0] = -1
                            row.update(calculate_transform_in_cell(metric='dist', array=dist_mask, bin=False, seg_id=seg_id,
                                                                   region='cell', mode='total', dist_norm=dist_norm))

                            # Calculate distance transform on cell per probe_object
                            dist_per_obj = []
                            for probe in probe_centers:
                                dist_obj = dist_mask[probe]
                                dist_per_obj.append(dist_obj)
                            row.update({seg_id + '_dist_per_obj':dist_per_obj})
                            dist_per_obj = np.array(dist_per_obj)
                            row.update(calculate_transform_in_cell(metric='dist', array=dist_per_obj, bin=True,
                                                                   dist_norm=dist_norm, seg_id=seg_id, region='cell',
                                                                   mode='per_obj'))

                            # Map probe on radon transform on cell per probe_area
                            radon_probe_mask = radon_org_mask * probe_in_mask
                            radon_probe_mask[radon_probe_mask==0] = -1
                            row.update(calculate_transform_in_cell(metric='radon', array=radon_probe_mask, bin=False,
                                                                   seg_id=seg_id, region='cell', mode='total', dist_norm=radon_probe_mask))

                            # Calculate radon transform on cell per probe_object
                            radon_per_obj = []
                            for probe in probe_centers:
                                radon_obj = radon_probe_mask[probe]
                                radon_per_obj.append(radon_obj)
                            row.update({seg_id + '_radon_per_obj': radon_per_obj})
                            radon_per_obj = np.array(radon_per_obj)
                            row.update(calculate_transform_in_cell(metric='radon', array=radon_per_obj, bin=True,
                                                                   dist_norm=radon_probe_mask, seg_id=seg_id, region='cell',
                                                                   mode='per_obj'))
                            #print ('radon')
                            probe_center_box = []
                            label_probe = measure.label(boxed_masks[seg_id])
                            props = regionprops(label_probe)
                            for probe in range (0, len(props)):
                                y = int(props[probe].centroid[0])
                                x = int(props[probe].centroid[1])
                                probe_center_box.append((y, x))
                            row.update({seg_id + '_probe_center_boxed': probe_center_box})

                            #print ('dist')
                            dist_nuc = calculate_distance_with_obj_seed(obj_seed=boxed_masks['nuc'], obj_ref_reverse=boxed_masks['cell_reverse'],
                                                                        set_coor=probe_center_box)
                            row.update({seg_id + '_dist_with_nuc_cell': dist_nuc})
                            dist_nuc = np.array(dist_nuc)
                            row.update(calculate_transform_in_cell(metric='dist_nuc', array=dist_nuc, bin=False, seg_id=seg_id,
                                                                   region='cell', mode='per_obj', dist_norm=dist_nuc))

                            dist_struc = calculate_distance_with_obj_seed(obj_seed=boxed_masks['struc'], obj_ref_reverse=boxed_masks['cell_reverse'],
                                                                          set_coor=probe_center_box)
                            row.update({seg_id + '_dist_with_struc_cell': dist_struc})
                            dist_struc = np.array(dist_struc)
                            row.update(calculate_transform_in_cell(metric='dist_struc', array=dist_struc, bin=False, seg_id=seg_id,
                                                                   region='cell', mode='per_obj', dist_norm=dist_struc))
                            #print ('abs dist')
                            # Calculate distance from nucleus in px
                            abs_dist_nuc = []
                            for probe in probe_center_box:
                                dist = nuc_rev_dt[probe[0], probe[1]]
                                abs_dist_nuc.append(dist)
                            row.update({seg_id + '_abs_dist_nuc': abs_dist_nuc})
                            abs_dist_nuc = np.array(abs_dist_nuc)
                            row.update(calculate_transform_in_cell(metric='abs_dist_nuc', array=abs_dist_nuc, bin=False, seg_id=seg_id,
                                                                   region='cell', mode='total', dist_norm=abs_dist_nuc))
                            print ('here')
                            # Calculate distance from nearest structure in px
                            abs_dist_struc = []
                            for probe in probe_center_box:
                                dist = struc_rev_dt[probe[0], probe[1]]
                                abs_dist_struc.append(dist)
                            row.update({seg_id + '_abs_dist_struc': abs_dist_struc})
                            abs_dist_struc = np.array(abs_dist_struc)
                            row.update(calculate_transform_in_cell(metric='abs_dist_struc', array=abs_dist_struc, bin=False,
                                                                   seg_id=seg_id, region='cell', mode='total', dist_norm=abs_dist_struc))

                    except:
                        print ('cannot process cell ' + str(cell_object) + ' in ' + seg_file)
                        df_failed_cell = df_failed_cell.append({'img': seg_file, 'cell': cell_object}, ignore_index=True)
                        pass
                    df = df.append(row, ignore_index=True)

failed = pd.DataFrame(failed_img)
df_failed_cell.to_csv(os.path.join(df_output_folder, 'failed_processing.csv'))
df.to_csv(os.path.join(df_output_folder, 'dist_radon_transform_all.csv'), ignore_index=True)


















