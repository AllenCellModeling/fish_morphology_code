#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "codecov",
    "black",
    "flake8",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

setup_requirements = ["pytest-runner"]

dev_requirements = [
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "flake8>=3.7.7",
    "ipython>=7.5.0",
    "m2r>=0.2.1",
    "pre-commit>=1.20.0",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "Sphinx>=2.0.0b1",
    "sphinx_rtd_theme>=0.1.2",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

interactive_requirements = ["altair", "jupyterlab", "matplotlib"]

requirements = [
    "aicsimageio>=3.0.2",
    "anndata>=0.6.22",
    "fire>=0.2.1",
    "imageio>=2.6.1",
    "numpy>=1.17.2",
    "quilt3distribute>=0.1.2",
    "pandas>=0.25.1",
    "scikit-image>=0.16.1",
    "scikit-learn>=0.21.3",
    "tqdm>=4.36.1",
]

extra_requirements = {
    "test": test_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "interactive": interactive_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements,
    ],
}

setup(
    author="Rory Donovan-Maiye",
    author_email="rorydm@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Allen Institute Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="data ingestion, processing, and analysis for cardio/FISH project",
    entry_points={
        "console_scripts": [
            "download_2D_segs=fish_morphology_code.bin.download_quilt_data:main_segs",
            "download_2D_contrasted=fish_morphology_code.bin.download_quilt_data:main_contrasted",
            "download_2D_features=fish_morphology_code.bin.download_quilt_data:main_features",
            "download_scrnaseq=fish_morphology_code.bin.download_quilt_data:main_scrnaseq",
            "contrast_and_segment=fish_morphology_code.bin.stretch:main",
            "merge_cellprofiler_output=fish_morphology_code.bin.merge_cellprofiler_output:main",
        ]
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="fish_morphology_code",
    name="fish_morphology_code",
    packages=find_packages(),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="fish_morphology_code/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/fish_morphology_code",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
