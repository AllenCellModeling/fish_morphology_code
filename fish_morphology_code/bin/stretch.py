#!/usr/bin/env python
"""
Extract hand segmented images of individual cells for scoring actn2 structure
"""

import json
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import fire
import pandas as pd
from tqdm import tqdm

from fish_morphology_code.processing.auto_contrast.stretch_utils import (
    field_worker,
    DEFAULT_CONTRAST_KWARGS,
    DEFAULT_CHANNEL_GROUPS,
)


def run(
    image_file_csv,
    channels_json,
    relative_paths=True,
    out_dir=None,
    image_dims="CYX",
    contrast_method="simple_quantile",
    contrast_kwargs=DEFAULT_CONTRAST_KWARGS,
    channel_groups=DEFAULT_CHANNEL_GROUPS,
    max_workers=None,
    verbose=False,
):
    """
    Autocontrast by field/channel, extract segmented objects from field, save output and log what happened
    Args:
        image_file_csv (str): csv file with list *absolute_paths* of max projects + seg file tiffs
        channels_json (str): json file with channel identifiers {"name": index}
        relative_paths (bool): if True, paths are interpreted as relative to the location of image_file_csv, default=True
        out_dir (str): where to save output images, default=None
        image_dims (str): input image dimension ordering, default="CYX"
        contrast_method (str): method for autocontrasting, default=="simple_quantile"
        contrast_kwargs (dict): keyword args for autocontrast settings,
            default={"fluor": {"clip_quantiles": [0.0, 0.998], "zero_below_median": False},
                     "bf": {"clip_quantiles": [0.00001, 0.99999], "zero_below_median": False},
                     "seg": {}}
        channel_groups (dict): fluor/bf/seg grouping,
            default={"fluor": ["488", "561", "638", "nuc"],
                     "bf": ["bf"],
                     "seg": ["seg488", "seg561", "seg638", "backmask", "cell"]}
        max_workers (int): how many jobs to run in parallel / cores to use. default=None, which uses all available cores.
        verbose (bool): print info while processing or not, default=False (True probably messes with the progess bar)
    """

    # save run parameters
    run_parameters = locals()

    # make output dir if doesn't exist yet
    # set out dir if needed
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)
    out_dir = out_dir.absolute()
    out_dir.mkdir(exist_ok=True)

    # write run parameters
    with open(out_dir.joinpath("parameters.json"), "w") as fp:
        json.dump(run_parameters, fp, indent=2, sort_keys=True)

    # read channel defs
    with open(channels_json, "r") as f:
        channels = json.load(f)

    # read input file manifest
    input_files = pd.read_csv(image_file_csv)
    file_names = [Path(f) for f in input_files["2D_fov_tiff_path"]]
    if relative_paths:
        file_names = [Path(image_file_csv).parent / f for f in file_names]

    # print task info
    if verbose:
        print(f"found {len(file_names)} image fields -- beginging processing")

    # partial function for iterating through files with map
    _field_worker_partial = partial(
        field_worker,
        image_dims=image_dims,
        out_dir=out_dir,
        contrast_method=contrast_method,
        contrast_kwargs=contrast_kwargs,
        channels=channels,
        channel_groups=channel_groups,
        verbose=verbose,
    )

    # iterate in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        main_log_df = pd.concat(
            tqdm(
                executor.map(_field_worker_partial, file_names),
                total=len(file_names),
                desc="Fields",
            ),
            axis="rows",
            ignore_index=True,
            sort=False,
        )

    # reorder dataframe columns
    ordered_cols = [
        "2D_fov_tiff_path",
        "rescaled_2D_fov_tiff_path",
        "cell_label_value",
        "rescaled_2D_single_cell_tiff_path",
    ]
    unordered_cols = [c for c in main_log_df.columns if c not in ordered_cols]
    cols = ordered_cols + unordered_cols
    main_log_df = main_log_df[cols]

    # if relative paths, reset
    if relative_paths:
        main_log_df["2D_fov_tiff_path"] = main_log_df["2D_fov_tiff_path"].apply(
            lambda p: str(Path(p).relative_to(Path(p).parent.parent))
        )

    # merge with input metadata
    main_log_df = input_files.merge(main_log_df)

    # write out csv
    main_log_df.to_csv(out_dir.joinpath("output_image_manifest.csv"), index=False)


def main():
    fire.Fire(run)
