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
pip install pre-commit
pre-commit install
```

This will install the black formatter and flake8 linter, and configure them to run as pre-commit hooks.

### Versions
**Stable Release:** `pip install fish_morphology_code`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/fish_morphology_code.git`

## Documentation
For full package documentation please visit [AllenCellModeling.github.io/fish_morphology_code](https://AllenCellModeling.github.io/fish_morphology_code).


## Running the code

To run the image normalization code, from the main repo dir:
```
python fish_morphology_code/processing/auto_contrast/stretch.py data/input_segs_and_tiffs/raw_seg_013_014_images.csv data/channel_defs.json --out_dir=/allen/aics/modeling/data/cardio_fish/normalized_2D_tiffs
```

## Running the tests

### Image normalization

To run with mostly default settings, from `fish_morphology_code/tests/` (only with access to `/allen/aics`):
```
./test.sh
```

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

    This will run `tox` which will run all your tests in both Python 3.6 and Python 3.7 as well as linting your code.

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

