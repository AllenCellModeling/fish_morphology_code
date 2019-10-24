import pandas as pd
import fire
from quilt3distribute import Dataset
from quilt3distribute.validation import validate


def distribute_seg_dataset(
    csv_loc="data/input_segs_and_tiffs/raw_seg_013_014_images.csv",
    col_name_map={
        "fov_path": "original_fov_location",
        "FOVId": "fov_id",
        "seg_file_name": "2D_tiff_path",
    },
    dataset_name="2d_segmented_fields",
    package_owner="rorydm",
    s3_bucket="s3://allencell-internal-quilt",
):

    # read in original csv
    df = pd.read_csv(csv_loc)

    # rename some cols
    df = df.rename(col_name_map, axis="columns")

    # drop any cols with missing data
    vds = validate(df, drop_on_error=True)
    df = vds.data.reset_index(drop=True)

    # create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="data/README.md",
    )

    # set data path cols, metadata cols, and extra files
    ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(["2D_tiff_path"])
    ds.set_extra_files(["data/channel_defs.json"])

    ds.distribute(s3_bucket)


if __name__ == "__main__":
    fire.Fire(distribute_seg_dataset)
