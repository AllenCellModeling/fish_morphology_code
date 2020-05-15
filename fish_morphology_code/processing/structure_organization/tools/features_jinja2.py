import os
import jinja2
import subprocess
import pandas as pd
from quilt3 import Package

# Downlaod the datasets from Quilt if there is no local copy

ds_folder = "../database/"

if not os.path.exists(os.path.join(ds_folder, "metadata.csv")):

    pkg = Package.browse(
        "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
    ).fetch(ds_folder)

metadata = pd.read_csv(os.path.join(ds_folder, "metadata.csv"))

df_fov = pd.read_csv(os.path.join(ds_folder, metadata.database_path[0]), index_col=0)


#
# Load jinja template
#

j2template = jinja2.Environment(loader=jinja2.FileSystemLoader(".")).get_template(
    "features.j2"
)

rfolder = (
    script_folder
) = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/fish_morphology_code/fish_morphology_code/processing/structure_organization/jinja2/"

for index in df_fov.index:

    render_dict = {"index": index}

    script_name = os.path.join(rfolder, "scripts", "job_" + str(index) + ".script")

    with open(script_name, "w") as f:
        f.writelines(j2template.render(render_dict))

    submission = "sbatch " + script_name
    print("Submitting command: {}".format(submission))
    process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
