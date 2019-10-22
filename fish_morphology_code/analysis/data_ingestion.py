import itertools

import numpy as np
import pandas as pd

COLUMN_GROUPS = {
    "score": ["mh_score", "kg_score"],
    "required_feats": ["nuc_AreaShape_Area", "cell_napari_AreaShape_Area"],
    "metadata": [
        "ImageNumber",
        "FileName_bf",
        "plate_id",
        "well_id",
        "probe_561",
        "probe_638",
        "cell_age",
        "nuc_ImageNumber",
        "nuc_ObjectNumber",
        "cell_napari_Number_Object_Number",
        "cell_napari_Parent_RelabeledNuclei",
        "nuc_Number_Object_Number",
        "nuc_Parent_FilterNuc",
        "nuc_Parent_napari_cell",
    ],
}


def load_data(csv_loc, index_col=0, low_memory=False, nrows=10067):
    """loads df from csv with some options"""
    df = pd.read_csv(csv_loc, index_col=index_col, low_memory=low_memory, nrows=nrows)
    return df


def prune_data(
    df,
    drop_cols=[
        "probe_488",
        "nuc_AreaShape_Center_Z",
        "cell_napari_AreaShape_Center_Z",
        "cell_napari_AreaShape_EulerNumber",
    ],
):
    """drop some columns"""
    return df.drop(drop_cols, axis="columns")


def drop_rows_with_nans(df, nonan_cols=list(itertools.chain(*COLUMN_GROUPS.values()))):
    """drop rows with values missing in those cols"""
    return df.dropna(subset=nonan_cols, axis="rows").reset_index(drop=True)


def drop_bad_struct_scores(df, exclude_scores=[-1, 0]):
    """-1 is a bad cell and 0 was non expressing i think?"""
    return df[
        (~df[COLUMN_GROUPS["score"]].isin(exclude_scores)).all(axis="columns")
    ].reset_index(drop=True)


def get_probe_pairs(df, probe_name_cols=["probe_561", "probe_638"]):
    """unique probe pairs in dataset"""
    return list(list(x) for x in df[probe_name_cols].drop_duplicates().values)


def clean_up_loc_scores(
    df,
    probe_loc_cols=["probe_561_loc_score", "probe_638_loc_score"],
    ints=["0", "1", "2", "3"],
    nans=[" ", "b", "n", "N", np.nan],
    threes2twos=True,
):
    """
    look at probe_loc_cols and map some scores to nans and others to ints
    threes2twos sets threes to twos, which becky and kaytlyn advised
    """

    loc_scores = list(
        set(x for l in [list(df[col].unique()) for col in probe_loc_cols] for x in l)
    )

    d_nan = {k: np.nan for k in loc_scores if k in nans}
    d_int = {k: int(k) for k in loc_scores if k in ints}
    if threes2twos:
        d_int["3"] = 2  # becky said ignore 3s and call them 2s
    d_map = {**d_nan, **d_int}

    df_out = df.copy()
    for col in probe_loc_cols:
        df_out[col] = df[col].map(d_map)

    return df_out
