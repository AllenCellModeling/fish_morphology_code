import fire
import subprocess
import pandas as pd
from quilt3distribute import Dataset


def distribute_probe_loc_dataset(
    test=False,
    csv_loc="sarc_classification.csv",
    col_name_map={
        "fov_path": "original_fov_location",
        "cell_num": "napariCell_ObjectNumber",
    },
    dataset_name="probe_structure_classifier",
    package_owner="calystay",
    s3_bucket="s3://allencell-internal-quilt",
    readme_path="README.md",
):

    df = pd.read_csv(csv_loc)
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "RawPath"])
    df = df.rename(columns=col_name_map)

    if test:
        df = df.loc[0:2]
        dataset_name = f"{dataset_name}_test"

    ds = Dataset(
        dataset=df,
        name=dataset_name,
        package_owner=package_owner,
        readme_path=readme_path,
    )

    ds.set_metadata_columns(["original_fov_location"])
    ds.set_path_columns(["nuc_mask_path"])

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket,
        message=f"mRNA probe location masked by sarcomere organization class.  Generated at git commit hash of fish_morphology_code = {label}",
    )


if __name__ == "__main__":
    fire.Fire(distribute_probe_loc_dataset)
