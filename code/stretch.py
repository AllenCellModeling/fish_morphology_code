#!/usr/bin/env python
"""
Extract hand segmented images of individual cells for scoring actn2 structure
"""

import os
import json
import fire
import pandas as pd
from tqdm import tqdm

from stretch_utils import stretch_worker


def run(
    image_file_csv,
    channels_json,
    out_dir=None,
    contrast_method="simple_quantile",
    auto_contrast_kwargs={
        "fluor_channels": ["488", "561", "638", "nuc"],
        "bf_channels": ["bf"],
        "fluor_kwargs": {"clip_quantiles": [0.0, 0.998], "zero_below_median": False},
        "bf_kwargs": {"clip_quantiles": [0.00001, 0.99999], "zero_below_median": False},
    },
    verbose=False,
):
    """
    Extract segmented objects from field of view and save channels into separate images

    image_file_csv: csv file with list *absolute_paths* of max projects + seg file tiffs
    channels_json: json file with channel identifiers {"name": index}
    out_dir: output directory where to save images and log, default=current working directory
    auto_contrast_kwargs: which channels to treat as fluroescent vs brightfield vs leave alone, and how to stretch those you wan to adjust
        default = {"fluor_channels":["488", "561", "638", "nuc"],
                   "bf_channels":["bf"],
                   "fluor_kwargs":{"clip_quantiles":[0.0,0.998], "zero_below_median":False},
                   "bf_kwargs":{"clip_quantiles":[0.00001, 0.99999], "zero_below_median":False}}
    """

    # save run parameters
    run_parameters = locals()

    # make output dir if doesn't exist yet
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # write run parameters
    with open(os.path.join(out_dir, "parameters.json"), "w") as fp:
        json.dump(run_parameters, fp, indent=2, sort_keys=True)

    # read channel defs
    with open(channels_json, "r") as f:
        channels = json.load(f)

    # read input file manifest
    input_files = pd.read_csv(image_file_csv)
    file_names = input_files["image_location"]

    # print task info
    if verbose:
        print("found {} image fields -- beginging processing".format(len(file_names)))

    # for each field, auto-contrast the channels if appropriate, and save single cell segmentations
    field_info_dfs = []
    for filename in tqdm(file_names, desc="Field"):
        field_info_df = stretch_worker(
            filename,
            out_dir=out_dir,
            channels=channels,
            contrast_method=contrast_method,
            verbose=verbose,
            auto_contrast_kwargs=auto_contrast_kwargs,
        )
        field_info_dfs += [field_info_df]

    # aggregate all the metadata for each single cell+channel image and save
    main_log_df = pd.concat(field_info_dfs, axis="rows", ignore_index=True)
    ordered_cols = [
        "field_image_name",
        "field_image_path",
        "rescaled_field_image_path",
        "channel_name",
        "channel_index",
        "cell_index",
        "cell_label_value",
        "single_cell_channel_output_path",
    ]
    unordered_cols = [c for c in main_log_df.columns if c not in ordered_cols]
    cols = ordered_cols + unordered_cols
    main_log_df = main_log_df[cols]
    main_log_df.to_csv(os.path.join(out_dir, "output_image_manifest.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(run)
