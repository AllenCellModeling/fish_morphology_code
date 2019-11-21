import subprocess
import fire
from quilt3distribute import Dataset


def distribute_classifier_features(
    test=False,
    csv_loc="classifier_features2quilt.csv",
    col_name_map={},
    dataset_name="structure_classifier_features",
    package_owner="matheusv",
    s3_bucket="s3://allencell-internal-quilt",
):

    # Create the dataset
    ds = Dataset(
        dataset=csv_loc,
        name=dataset_name,
        package_owner=package_owner,
        readme_path="README.md",
    )

    # Optionally add common additional requirements
    ds.add_usage_doc("https://docs.quiltdata.com/walkthrough/reading-from-a-package")
    ds.add_license("https://www.allencell.org/terms-of-use.html")

    # Optionally indicate column values to use for file metadata
    ds.set_metadata_columns(["cell_line", "structure"])

    # Optionally rename the columns on the package level
    ds.set_column_names_map({"feature_file": "structure_classifier_features"})

    # add commit hash to message
    label = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    # Distribute
    ds.distribute(
        push_uri=s3_bucket, message=f"git commit hash of fish_morphology_code = {label}"
    )


if __name__ == "__main__":
    fire.Fire(distribute_classifier_features)
