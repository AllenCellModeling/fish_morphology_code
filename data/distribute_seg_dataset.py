import subprocess
import pandas as pd
import fire
from quilt3distribute import Dataset
from quilt3distribute.validation import validate


def distribute_seg_dataset(
    csv_loc="input_segs_and_tiffs/raw_seg_013_014_images.csv",
    n_subsamples=None,
    col_name_map={
        "fov_path": "original_fov_location",
        "FOVId": "fov_id",
        "seg_file_name": "2D_fov_tiff_path",
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

    # subsample df for eg a test dataset
    if n_subsamples is not None:
        df = df.sample(n_subsamples)

    # create the dataset
    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="README.md",
    )

    # structure scores as auxilary file
    score_files = [
        f"structure_scores/structure_score_55000000{p}.csv" for p in (13, 14)
    ]
    score_dfs = [
        pd.read_csv(f).rename({"mh Score": "mh score"}, axis="columns")
        for f in score_files
    ]
    df_score = pd.concat(score_dfs, axis="rows", ignore_index=True, sort=False)
    df_score.to_csv("structure_scores.csv")

    # set data path cols, metadata cols, and extra files
    ds.set_metadata_columns(["fov_id", "original_fov_location"])
    ds.set_path_columns(["2D_fov_tiff_path"])
    ds.set_extra_files(["channel_defs.json", "structure_scores.csv"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_seg_dataset)
