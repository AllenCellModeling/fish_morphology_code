import pandas as pd
from quilt3distribute import Dataset

# Read dataset
df = pd.read_csv("../../fish_morphology_code/processing/structure_organization/results/AssayDevFishAnalsysis-Handoff.csv")

# Define package
ds = Dataset(
    dataset = df,
    name = "assay_dev_fish_analysis",
    package_owner = "matheus",
    readme_path = "../../fish_morphology_code/processing/structure_organization/tools/assay-dev-fish.md"
)

# Metadata
ds.set_metadata_columns(["CellId"])
ds.set_path_columns(['result_image_path'])

# Send to Quilt
pkg = ds.distribute(push_uri="s3://allencell-internal-quilt", message="Fish dataset by assay-dev")

# Distribute a test version as well
df = df.sample(n=1)

# Define package
ds = Dataset(
    dataset = df,
    name = "assay_dev_fish_analysis_test",
    package_owner = "matheus",
    readme_path = "../../fish_morphology_code/processing/structure_organization/tools/assay-dev-fish.md"
)

# Metadata
ds.set_metadata_columns(["CellId"])
ds.set_path_columns(['result_image_path'])

# Send to Quilt
pkg = ds.distribute(push_uri="s3://allencell-internal-quilt", message="Fish test dataset by assay-dev")
