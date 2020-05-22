import os
import pandas as pd
from tqdm import tqdm
from quilt3distribute import Dataset
from quilt3 import Package

# Downlaod the datasets from Quilt if there is no local copy

ds_folder = "../database/"

if not os.path.exists(os.path.join(ds_folder, "metadata.csv")):

    pkg = Package.browse(
        "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
    ).fetch(ds_folder)

metadata = pd.read_csv(os.path.join(ds_folder, "metadata.csv"))

df_meta = pd.read_csv(os.path.join(ds_folder, metadata.database_path[0]), index_col=0)

# Gathering results
df = []
for FOVId in tqdm(df_meta.index):

    prefix = os.path.join('..', 'output', f"fov_{FOVId}")

    if os.path.exists(prefix + ".csv"):

        df_fov = pd.read_csv(prefix + ".csv")

        df_fov["FOVId"] = FOVId
        df_fov["result_image_path"] = os.path.abspath(prefix + ".tif")
        df_fov["original_fov_location"] = df_meta.RawFileName[FOVId].replace("_C0.tif", "")

        df.append(df_fov)

    else:

        print(f"Data for FOV {FOVId} not found")

df = pd.concat(df, axis=0, sort=True).reset_index(drop=True)

# Adding new columns
for index in tqdm(df.index):
    FOVId = df.FOVId[index]
    CellId = df.CellId[index]
    df.loc[index, "napariCell_ObjectNumber"] = CellId
    df.loc[index, "NewCellId"] = f"fov-{FOVId}-cell-{CellId}"

# Dropping columnsn no longer needed and renaming the index column
df = df.drop(["FOVId", "CellId"], axis=1).rename(columns={"NewCellId": "CellId"})
print(df.head())

df["structure_name"] = "ACTN2"

# Save CSV for assay-dev internal analysis
if not os.path.exists("../results/"):
    os.makedirs("../results/")
df.set_index(["CellId"]).to_csv("../results/AssayDevFishAnalsysis2020.csv")

# -------------------------------------------------------------------------------------------------
# Upload results to Quilt
# -------------------------------------------------------------------------------------------------

# Information for Readme file that goes to Quilt
metadata = [
    {
        "CellId": {
            "name": None,
            "description": "Unique id that indentifies the FOV and the cell mask label",
        }
    },
    {
        "napariCell_ObjectNumber": {
            "name": None,
            "description": "Unique id that indentifies the label of cell segmentation in the fov",
        }
    },
    {"Age": {"name": None, "description": "Cells age"}},
    {
        "result_image_path": {
            "name": None,
            "description": "Z Stack with data produced by assay-dev",
        }
    },
    {"original_fov_location": {"name": None, "description": "Path to raw data"}},
    {"Total_Area": {"name": None, "description": "Number of pixels in cell mask"}},
    {
        "Frac_Area_Background": {
            "name": None,
            "description": "Fraction of cell area classified as background",
        }
    },
    {
        "Frac_Area_DiffuseOthers": {
            "name": None,
            "description": "Fraction of cell area classified as diffuse and others",
        }
    },
    {
        "Frac_Area_Fibers": {
            "name": None,
            "description": "Fraction of cell area classified as fibers",
        }
    },
    {
        "Frac_Area_Disorganized_Puncta": {
            "name": None,
            "description": "Fraction of cell area classified as disorganized puncta",
        }
    },
    {
        "Frac_Area_Organized_Puncta": {
            "name": None,
            "description": "Fraction of cell area classified as organized puncta",
        }
    },
    {
        "Frac_Area_Organized_ZDisks": {
            "name": None,
            "description": "Fraction of cell area classified as organized z disks",
        }
    },
    {
        "Prob_DiffuseOthers": {
            "name": None,
            "description": "Average probability of a pixel inside the cell to be classified as diffuse and others",
        }
    },
    {
        "Prob_Fibers": {
            "name": None,
            "description": "Average probability of a pixel inside the cell to be classified as fibers",
        }
    },
    {
        "Prob_Disorganized_Puncta": {
            "name": None,
            "description": "Average probability of a pixel inside the cell to be classified as disorganized puncta",
        }
    },
    {
        "Prob_Organized_Puncta": {
            "name": None,
            "description": "Average probability of a pixel inside the cell to be classified as organized puncta",
        }
    },
    {
        "Prob_Organized_ZDisks": {
            "name": None,
            "description": "Average probability of a pixel inside the cell to be classified as organized z disks",
        }
    },
    {
        "Intensity_Median": {
            "name": "IntensityMedian",
            "description": "Median of GFP signal in cell mask",
        }
    },
    {
        "Intensity_Integrated": {
            "name": "IntensityIntegrated",
            "description": "Integrated GFP signal in cell mask",
        }
    },
    {
        "Intensity_Median_BackSub": {
            "name": "IntensityMedianBkgSub",
            "description": "Median of GFP signal in cell mask with background subtracted (10% percentile",
        }
    },
    {
        "Intensity_Integrated_BackSub": {
            "name": "IntensityIntegratedBkgSub",
            "description": "Integrated GFP signal in cell mask with background subtracted (10% percentile",
        }
    },
    {
        "Maximum_Coefficient_Variation": {
            "name": None,
            "description": "Maximum value of the coefficient of variation obtained from correlation plots",
        }
    },
    {
        "Peak_Height": {
            "name": None,
            "description": "High of the highest peak in the correlation plots",
        }
    },
    {
        "Peak_Distance": {
            "name": None,
            "description": "Distance in pixels in which the maximum of the highest peak occurs",
        }
    },
    {
        "Peak_Angle": {
            "name": None,
            "description": "Angle in degrees for which we observe the highest correlation value",
        }
    },
]

# Selecting features from dataframe according to metadata above
selected_features = [key for f in metadata for key, _ in f.items()]

df = df[selected_features]

# Rename features according to metadata
features_rename = [
    {key: value["name"]}
    for f in metadata
    for key, value in f.items()
    if value["name"] is not None
]

for feature in features_rename:
    df = df.rename(columns=feature)

with open("assay-dev-fish.md", "w") as ftxt:
    ftxt.write("### Global structure organization and local structural alignment features\n\n")
    for meta in metadata:
        for key, value in meta.items():
            ftxt.write(
                "- `{0}`: {1}\n".format(
                    value["name"] if value["name"] is not None else key,
                    value["description"],
                )
            )

# Save a hand off version for the Modeling team
df.to_csv("../results/AssayDevFishAnalsysis-Handoff.csv")

# Upload to Quilt
ds = Dataset(
    dataset="../results/AssayDevFishAnalsysis-Handoff.csv",
    name="assay_dev_fish_analysis",
    package_owner="matheus",
    readme_path="assay-dev-fish.md",
)

ds.set_metadata_columns(["CellId"])
ds.set_path_columns(["result_image_path"])

# Send to Quilt
pkg = ds.distribute(
    push_uri="s3://allencell-internal-quilt", message="Fish dataset by assay-dev"
)
