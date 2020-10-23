import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from quilt3 import Package
from skimage import io as skio


def process_fov(FOVId, df_fov):

    """
        Takes a FOVId and the FOV dataframe and calculates local structure
        organization adn global strucuural alignment features for every single
        segmented cell in that FOV.
    """

    # Channel numbers
    ch_raw = 0 # highest intensity slice
    ch_sum = 1 # sum projection
    ch_prs = slice(2, 7) # probabilities
    ch_cla = 7 # classification maps
    ch_msk = 8 # cell segmentation masks

    source = "../output/"

    data = skio.imread(os.path.join(source, f"fov_{FOVId}.tif"))

    # Load global structural alignment features
    df_features = pd.read_csv(os.path.join(source, f"fov_{FOVId}.csv"), index_col=0)

    # df_features = df_features.set_index('CellId', drop=True)

    # Calculate local structure organization features and merge
    for CellId in tqdm(df_fov.index):

        # Get single cell masks
        mask = (data[ch_msk].astype(np.uint8) == CellId).astype(np.uint8)

        total_area = mask.sum()

        # Classification results from CNN classifier
        classification = data[ch_cla].astype(np.uint8)

        # Total number of classes is 5 + background (0)
        covered_area = np.bincount(classification[mask > 0].flatten(), minlength=6)

        covered_area = covered_area / total_area

        probs = data[ch_prs, mask > 0].mean(axis=1)

        # Intensity based features on highest intensity slice
        intensities = data[ch_raw, mask > 0].flatten()

        background_intensity = np.percentile(data[ch_raw], 1)

        avg_intensity = np.mean(intensities)

        med_intensity = np.percentile(intensities, 50)

        int_intensity = np.sum(intensities)

        avg_intensity_bs = np.mean(intensities - background_intensity)

        med_intensity_bs = np.percentile(intensities - background_intensity, 50)

        int_intensity_bs = np.sum(intensities - background_intensity)

        # Intensity based features on sum projection
        nslices = int(df_fov.at[CellId,'NSlices'])

        sum_intensities = data[ch_sum, mask > 0].flatten()

        int_sum_intensity = np.sum(sum_intensities)

        int_sum_intensity_bs = np.sum(sum_intensities - nslices*background_intensity)

        # Dict for features
        features = {
            "Total_Area": total_area,
            "Frac_Area_Background": covered_area[0],
            "Frac_Area_DiffuseOthers": covered_area[1],
            "Frac_Area_Fibers": covered_area[2],
            "Frac_Area_Disorganized_Puncta": covered_area[3],
            "Frac_Area_Organized_Puncta": covered_area[4],
            "Frac_Area_Organized_ZDisks": covered_area[5],
            "Prob_DiffuseOthers": probs[0],
            "Prob_Fibers": probs[1],
            "Prob_Disorganized_Puncta": probs[2],
            "Prob_Organized_Puncta": probs[3],
            "Prob_Organized_ZDisks": probs[4],
            "Intensity_Mean": avg_intensity,
            "Intensity_Median": med_intensity,
            "Intensity_Integrated": int_intensity,
            "Intensity_SumIntegrated": int_sum_intensity,
            "Intensity_Mean_BackSub": avg_intensity_bs,
            "Intensity_Median_BackSub": med_intensity_bs,
            "Intensity_Integrated_BackSub": int_intensity_bs,
            "Intensity_SumIntegrated_BackSub": int_sum_intensity_bs,
            "Background_Value": background_intensity
        }

        for key, value in features.items():
            df_features.loc[CellId, key] = value

    cols_to_use = [c for c in df_features if c not in df_fov.columns]

    df = pd.merge(df_fov, df_features[cols_to_use], left_index=True, right_index=True)

    df.to_csv(os.path.join(source, f"fov_{FOVId}.csv"))

    print("DONE")


if __name__ == "__main__":

    # Get FOV id as argument
    parser = argparse.ArgumentParser(
        description="Merge local organization and global alignment features"
    )
    parser.add_argument("--fov", help="Full path to FOV", required=True)
    args = vars(parser.parse_args())
    FOVId = int(args["fov"])

    # Downlaod the datasets from Quilt if there is no local copy
    ds_folder = "../database/"

    if not os.path.exists(os.path.join(ds_folder, "metadata.csv")):

        pkg = Package.browse(
            "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
        ).fetch(ds_folder)

    metadata = pd.read_csv(os.path.join(ds_folder, "metadata.csv"))

    df_meta_fov = pd.read_csv(
        os.path.join(ds_folder, metadata.database_path[0]), index_col=1
    )

    df_meta_cell = pd.read_csv(os.path.join(ds_folder, metadata.cell_database_path[0]))

    # Merge dataframes

    df_meta_cell["FOVId"] = df_meta_fov.loc[df_meta_cell.RawFileName].FOVId.values

    df_meta_cell = df_meta_cell.set_index(["FOVId", "CellId"])

    df_meta_cell = df_meta_cell.sort_index()

    # Run FOV
    process_fov(FOVId=FOVId, df_fov=df_meta_cell.loc[(FOVId,)])
