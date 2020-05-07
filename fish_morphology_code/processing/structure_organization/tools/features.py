import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io as skio
from skimage import measure as skmeasure
from scipy import stats as scistats

def ProcessFOV(FOVId, df_fov):

    '''
        Takes an FIVId and the FOV dataframe and calculates local structure
        organization adn global strucuural alignment features for every single
        segmented cell in that FOV.
    '''

    # Channel numbers
    ch_raw = 0
    ch_prs = slice(1,6)
    ch_cla = 6
    ch_msk = 7

    df_features = []

    source = '../output/'

    data = skio.imread(os.path.join(source,f'fov_{FOVId}.tif'))

    # Load global structural alignment features    
    df_align = pd.read_csv(os.path.join(source,f'fov_{FOVId}.csv'), index_col=0)

    # Calculate local structure organization features and merge
    for CellId in tqdm(df_fov.index):

        # Get single cell masks
        mask = (data[ch_msk].astype(np.uint8)==CellId).astype(np.uint8)

        total_area = mask.sum()

        # Classification results from CNN classifier
        classification = data[ch_cla].astype(np.uint8) - 1

        # Total number of classes is 5 + background (0)
        covered_area = np.bincount(classification[mask>0].flatten(), minlength=6)

        covered_area = covered_area / total_area

        probs = data[ch_prs,mask>0].mean(axis=1)

        # Intensity based features
        intensities = data[ch_raw,mask>0].flatten()
        
        background_intensity = np.percentile(data[ch_raw], 10)

        med_intensity = np.percentile(intensities, 50)

        int_intensity = np.sum(intensities)

        med_intensity_bs = np.percentile(intensities-background_intensity, 50)

        int_intensity_bs = np.sum(intensities-background_intensity)

        # Dict for features
        features = {
            'FOVId': FOVId,
            'CellId': CellId,
            'Total_Area': total_area,
            'Frac_Area_Background': covered_area[0],
            'Frac_Area_DiffuseOthers': covered_area[1],
            'Frac_Area_Fibers': covered_area[2],
            'Frac_Area_Disorganized_Puncta': covered_area[3],
            'Frac_Area_Organized_Puncta': covered_area[4],
            'Frac_Area_Organized_ZDisks': covered_area[5],
            'Prob_DiffuseOthers': probs[0],
            'Prob_Fibers': probs[1],
            'Prob_Disorganized_Puncta': probs[2],
            'Prob_Organized_Puncta': probs[3],
            'Prob_Organized_ZDisks': probs[4],
            'Intensity_Median': med_intensity,
            'Intensity_Integrated': int_intensity,
            'Intensity_Median_BackSub': med_intensity_bs,
            'Intensity_Integrated_BackSub': int_intensity_bs,
            'Background_Value': background_intensity
        }

        features.update(df_align.loc[CellId].to_dict())

        df_features.append(features)

    df_features = pd.DataFrame(df_features).set_index(['FOVId','CellId'])

    df = pd.merge(
        df_fov,
        df_features,
        left_index = True,
        right_index = True
    )

    df.to_csv(os.path.join(source,f'fov_{FOVId}.csv'))

    print('DONE')

if __name__ == "__main__":

    # Get FOV id as argument
    parser = argparse.ArgumentParser(description="Merge local organization and global alignment features")
    parser.add_argument("--fov", help="Full path to FOV", required=True)
    args = vars(parser.parse_args())   
    FOVId = int(args['fov'])

    # Gather necessary information
    df_fov = pd.read_csv('../database/database.csv', index_col=1)

    df_cell = pd.read_csv('../database/cell_database.csv')

    df_cell['FOVId'] = df_fov.loc[df_cell.RawFileName].FOVId.values

    df_cell = df_cell.set_index(['FOVId','CellId'])

    df_cell = df_cell.sort_index()

    # Run FOV
    ProcessFOV(FOVId=FOVId, df_fov=df_cell.loc[(FOVId,)])
