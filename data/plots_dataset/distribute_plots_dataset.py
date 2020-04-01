import subprocess
import fire
from quilt3distribute import Dataset

from fish_morphology_code.analysis.collate_plot_dataset import collate_plot_dataset


def manuscript_plots_dataset(
    test=False,
    col_name_map={},
    dataset_name="manuscript_plots",
    package_owner="rorydm",
    readme_path="README.md",
    s3_bucket="s3://allencell-internal-quilt",
):

    df, _ = collate_plot_dataset()

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

    # tag with commit hash
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    ds.distribute(
        s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(manuscript_plots_dataset)
