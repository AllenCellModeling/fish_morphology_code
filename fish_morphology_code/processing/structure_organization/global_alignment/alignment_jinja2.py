import os
import jinja2
import subprocess
import numpy as np
import pandas as pd

#
# Load database
#

database = pd.read_csv('../database/database.csv', index_col=0)

#
# Load jinja template
#

j2template = jinja2.Environment(loader=jinja2.FileSystemLoader('.')).get_template('alignment.j2')

rfolder = script_folder = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/fish_morphology_code/fish_morphology_code/processing/structure_organization/jinja2/'

for f in ['','scripts','log']:
        if not os.path.exists(os.path.join(rfolder,f)):
                os.makedirs(os.path.join(rfolder,f))

for index in database.index:

        render_dict = {
                'index': index
        }

        script_name = os.path.join(rfolder,'scripts','job_'+ str(index) +'.script')

        with open(script_name,'w') as f:
                f.writelines(j2template.render(render_dict))

        submission = "sbatch " + script_name
        print("Submitting command: {}".format(submission))
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
        (out, err) = process.communicate()

