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

    ch_raw = 0
    ch_seg = 1
    ch_bkg = 2
    ch_vor = 3
    ch_prs = slice(4,9)
    ch_cla = 9
    ch_msk = 10

    df_features = []

    source = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/output/'

    data = skio.imread(os.path.join(source,f'fov_{FOVId}.tif'))

    #
    # Orientation features
    #
    
    df_orient = pd.read_csv(os.path.join(source,f'fov_{FOVId}.orientation'), index_col=0)

    for CellId in tqdm(df_fov.index):

        mask = (data[ch_msk].astype(np.uint8)==CellId).astype(np.uint8)

        #
        # CNN classification features
        #

        classification = data[ch_cla].astype(np.uint8) - 1

        total_area = mask.sum()

        covered_area = np.bincount(classification[mask>0].flatten(), minlength=6)

        covered_area = covered_area / total_area

        probs = data[ch_prs,mask>0].mean(axis=1)

        #
        # Intensity based features
        #

        intensities = data[ch_raw,mask>0].flatten()
        
        background_intensity = np.percentile(data[ch_raw], 10)

        med_intensity = np.percentile(intensities, 50)

        int_intensity = np.sum(intensities)

        med_intensity_bs = np.percentile(intensities-background_intensity, 50)

        int_intensity_bs = np.sum(intensities-background_intensity)

        #
        # Dict for features
        #

        features = {
            'FOVId': FOVId,
            'CellId': CellId,
            'TotalArea': total_area,
            'FracAreaBackground': covered_area[0],
            'FracAreaMessy': covered_area[1],
            'FracAreaThreads': covered_area[2],
            'FracAreaRandom': covered_area[3],
            'FracAreaRegularDots': covered_area[4],
            'FracAreaRegularStripes': covered_area[5],
            'ProbMessy': probs[0],
            'ProbThreads': probs[1],
            'ProbRandom': probs[2],
            'ProbRegularDots': probs[3],
            'ProbRegularStripes': probs[4],
            'Intensity_Median': med_intensity,
            'Intensity_Integrated': int_intensity,
            'Intensity_Median_BackSub': med_intensity_bs,
            'Intensity_Integrated_BackSub': int_intensity_bs,
            'BackgroundValue': background_intensity
        }

        features.update(df_orient.loc[CellId].to_dict())

        df_features.append(features)

    df_features = pd.DataFrame(df_features).set_index(['FOVId','CellId'])

    pd.merge(
        df_fov,
        df_features,
        left_index = True,
        right_index = True
    ).to_csv(os.path.join(source,f'fov_{FOVId}.csv'))

    print('DONE')

if __name__ == "__main__":

    #
    # Get FOV id
    #

    parser = argparse.ArgumentParser(description="Runs Radon analysis on a particular FOV")
    parser.add_argument("--fov", help="Full path to FOV", required=True)
    args = vars(parser.parse_args())   

    #
    # Run FOV
    #

    df_fov = pd.read_csv('database/database.csv', index_col=1)

    df_cell = pd.read_csv('database/cell_database.csv')

    df_cell['FOVId'] = df_fov.loc[df_cell.RawFileName].FOVId.values

    df_cell = df_cell.set_index(['FOVId','CellId'])

    df_cell = df_cell.sort_index()

    FOVId = int(args['fov'])

    ProcessFOV(FOVId=FOVId, df_fov=df_cell.loc[(FOVId,)])
