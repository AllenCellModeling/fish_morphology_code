# Fish Morphology Code

[![Build Status](https://github.com/AllenCellModeling/fish_morphology_code/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/fish_morphology_code/actions)
[![Documentation](https://github.com/AllenCellModeling/fish_morphology_code/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/fish_morphology_code)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/fish_morphology_code/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/fish_morphology_code)

data ingestion, processing, and analysis for cardio/FISH project

---


## Installation

Recommended installation procedure is to create a conda environment and then pip install into that environment.

### Normal users

Clone this repository, then

```
cd fish_morphology_code
conda create --name cardio_fish_data python=3.7
conda activate cardio_fish_data
pip install -e .
```

### Developers

After following the "Normal users" installation instructions,

```
pip install -e .[all]
pip install pre-commit
pre-commit install
```

This will install the the dev dependencies.
It will also install the black formatter and flake8 linter, and configure them to run as pre-commit hooks.

### Versions
**Stable Release:** `pip install fish_morphology_code`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/fish_morphology_code.git`


## Documentation
For full package documentation please visit [AllenCellModeling.github.io/fish_morphology_code](https://AllenCellModeling.github.io/fish_morphology_code).


## Downloading the data

### Un-normalized 2D tiffs

The images in this dataset are 16 bits.

To download a test dataset (~40 MB) of just two fields:

```download_2D_segs --test=True```

This will download the dataset and save it to the `quilt_data_test` directory.

To get the whole dataset (~9 GB) of all 478 fields:

```download_2D_segs```

This will download the dataset and save it to the `quilt_data` directory.


### Normalized (autocontrasted) 2D tiffs

The images in this dataset are 8 bits.

To download a test dataset (~25 MB) of just two fields:

```download_2D_contrasted --test=True```

This will download the dataset and save it to the `2d_autocontrasted_fields_and_single_cells_test` directory.

To get the whole dataset (~7 GB) of all 478 fields:

```download_2D_contrasted```

This will download the dataset and save it to the `2d_autocontrasted_fields_and_single_cells` directory.


## Running the auto-contrasting code

To run the image normalization code, from the main repo dir:
```
contrast_and_segment quilt_data/metadata.csv quilt_data/supporting_files/channel_defs.json --out_dir=output_data
```

## Running the tests

To run the `pytest` tests defined in `fish_morphology_code/tests` via `tox`, use
```
make build
```

## Uploading data to quilt

Uploading to quilt should happen infrequently so all the code has been put in the `data` directory outside of the main `fish_morphology_code` library.

### Un-normalized 2D tiffs
To upload a new version of the un-normalized 2D fovs, from the `data` directory:
```python distribute_seg_dataset.py```

### Normalized 2D tiffs
To upload a new version of the normalized (autocontrasted) 2D fovs + single cell images, from the `data` directory:
```python distribute_autocontrasted_dataset.py```


## Bare minimum contents contains:

- [x] manifest of the locations of original image data + segmentations
- [x] script for taking original images and generating 2D normalized images
- [x] structure annotations for each segmented cell
- [ ] text-ish version of cellprofiler workflow to generate single cell segmentations and features from image data + segmentations


## Stretch goals

- [ ] original image data + annotations into fms/labkey/quilt, or somehow programatically downloadable


## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

#### The Three Commands You Need To Know
1. `make build`

    This will run `tox` which will run all your tests in Python 3.7 as well as linting your code.

2. `make clean`

    This will clean up various Python and build generated files so that you can ensure that you are working in a clean
    environment.

3. `make docs`

    This will generate and launch a web browser to view the most up-to-date documentation for your Python package.

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. GitHub
Actions will run on every push and on a CRON to this branch but still recommended to commit to your development
branches and make pull requests to master.
2. `stable` is for releases only. When you want to release your project on PyPI, simply make a PR from `master` to
`stable`, this template will handle the rest as long as you have added your PyPI information described in the above
**Optional Steps** section.
3. Your day-to-day work should exist on branches separate from `master`. Even if it is just yourself working on the
repository, make a PR from your working branch to `master` so that you can ensure your commits don't break the
development head. GitHub Actions will run on every push to any branch or any pull request from any branch to any other
branch.

#### Additional Optional Setup Steps:
* Register fish_morphology_code with Codecov:
  * Make an account on [codecov.io](https://codecov.io) (Recommended to sign in with GitHub)
  * Select `AllenCellModeling` and click: `Add new repository`
  * Copy the token provided, go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/AllenCellModeling/fish_morphology_code/settings/secrets),
  add a secret called `CODECOV_TOKEN` with the token you just copied.
  Don't worry, no one will see this token because it will be encrypted.
* Generate and add an access token as a secret to the repository for auto documentation generation to work
  * Go to your [GitHub account's Personal Access Tokens page](https://github.com/settings/tokens)
  * Click: `Generate new token`
  * _Recommendations:_
    * _Name the token: "Auto-Documentation Generation" or similar so you know what it is being used for later_
    * _Select only: `repo:status`, `repo_deployment`, and `public_repo` to limit what this token has access to_
  * Copy the newly generated token
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/AllenCellModeling/fish_morphology_code/settings/secrets),
  add a secret called `ACCESS_TOKEN` with the personal access token you just created.
  Don't worry, no one will see this password because it will be encrypted.
* Register your project with PyPI:
  * Make an account on [pypi.org](https://pypi.org)
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/AllenCellModeling/fish_morphology_code/settings/secrets),
  add a secret called `PYPI_TOKEN` with your password for your PyPI account.
  Don't worry, no one will see this password because it will be encrypted.
  * Next time you push to the branch: `stable`, GitHub actions will build and deploy your Python package to PyPI.
  * _Recommendation: Prior to pushing to `stable` it is recommended to install and run `bumpversion` as this will,
  tag a git commit for release and update the `setup.py` version number._
* Add branch protections to `master` and `stable`
    * To protect from just anyone pushing to `master` or `stable` (the branches with more tests and deploy
    configurations)
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/AllenCellModeling/fish_morphology_code/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require pull request reviews before merging_
      * _Require status checks to pass before merging (Recommended: lint and test)_
      * _Restrict who can push to matching branches_


***Free software: Allen Institute Software License***

