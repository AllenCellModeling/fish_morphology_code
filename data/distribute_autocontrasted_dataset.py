import subprocess
import pandas as pd
import fire
from quilt3distribute import Dataset
from quilt3distribute.validation import validate


def distribute_autocontrasted_dataset(
    test=False,
    csv_loc="../output_data/output_image_manifest.csv",
    col_name_map={},
    dataset_name="2d_autocontrasted_fields_and_single_cells",
    package_owner="rorydm",
    s3_bucket="s3://allencell-internal-quilt",
):

    # read in original csv
    df = pd.read_csv(csv_loc)

    # rename some cols
    df = df.rename(col_name_map, axis="columns")
    df = df.drop(["2D_fov_tiff_path"], axis="columns").rename(
        col_name_map, axis="columns"
    )

    # drop any cols with missing data
    vds = validate(df, drop_on_error=True)
    df = vds.data.reset_index(drop=True)

    # subsample df for eg a test dataset
    if test:
        df = df.sample(2, random_state=0)
        dataset_name = f"{dataset_name}_test"

    # create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="README.md",
    )

    # set data path cols, metadata cols, and extra files
    ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(
        ["rescaled_2D_fov_tiff_path", "rescaled_2D_single_cell_tiff_path"]
    )
    ds.set_extra_files(["channel_defs.json", "../output_data/parameters.json"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_autocontrasted_dataset)
