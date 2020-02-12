from pathlib import Path
import subprocess
import pandas as pd
import fire
from quilt3distribute import Dataset
from quilt3distribute.validation import validate


def distribute_nuclear_masks(
    test=False,
    csv_loc=Path("my/path/to/input/data/manifest.csv"),
    dataset_name="2d_nuclear_masks",
    package_owner="calystay",
    s3_bucket="s3://allencell-internal-quilt",
    readme_path="README.md",
):

    # read in original csv
    df_in = pd.read_csv(csv_loc)

    # make your df that has paths to nuclear masks
    df = pd.DataFrame(columns=["fov_id", "original_fov_location", "nuclear_mask_path"])

    # or maybe ??? just an idea
    df = df_in.merge(df)

    # TODO: fill in this df with the right paths / ids

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
        readme_path=readme_path,
    )

    # set data path cols, metadata cols, and extra files
    ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(["nuclear_mask_path"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_nuclear_masks)
