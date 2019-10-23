import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

COLUMN_GROUPS = {
    "struct_score": ["mh_score", "kg_score"],
    "loc_score": ["probe_561_loc_score", "probe_638_loc_score"],
    "required_feats": [
        "nuc_AreaShape_Area",
        "cell_napari_AreaShape_Area",
        "nuc_Children_seg_probe_561_Count",
        "nuc_Children_seg_probe_638_Count",
        "cell_napari_Children_seg_probe_561_Count",
        "cell_napari_Children_seg_probe_638_Count",
    ],
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
    """drop some columns that are mostly nans / useless"""
    return df.drop(drop_cols, axis="columns")


def drop_rows_with_nans(
    df,
    nonan_cols=list(
        itertools.chain(
            [e for k, v in COLUMN_GROUPS.items() if k != "loc_score" for e in v]
        )
    ),
):
    """drop rows with values missing in those cols"""
    return df.dropna(subset=nonan_cols, axis="rows").reset_index(drop=True)


def drop_bad_struct_scores(df, exclude_scores=[-1, 0]):
    """-1 is a bad cell and 0 was non expressing i think?"""
    return df[
        (~df[COLUMN_GROUPS["struct_score"]].isin(exclude_scores)).all(axis="columns")
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


def remap_struct_scores(
    df,
    struct_score_cols=COLUMN_GROUPS["struct_score"],
    map_dict={1: 1, 2: 1, 3: 2, 4: 3, 5: 3},
):
    """remap structure scores from 1:5 to 1:3 or whatever"""
    df_out = df.copy()
    for col in struct_score_cols:
        df_out[col] = df_out[col].map(map_dict)
    return df_out


def subset_to_probe_pair(df, probe_pair=["MYH6", "MYH7"]):
    """subset the df to only a certain probe pair"""
    return df[np.all(df[["probe_561", "probe_638"]] == probe_pair, axis=1)].reset_index(
        drop=True
    )


def get_feat_col_groups(df, prune_loc_feats=True):
    """get dict of col names corresponding to different types of features"""

    all_non_metadata_cols = [
        c for c in df.columns if c not in COLUMN_GROUPS["metadata"]
    ]

    feat_cols = {
        "probe_561": [c for c in all_non_metadata_cols if "probe_561" in c],
        "probe_638": [c for c in all_non_metadata_cols if "probe_638" in c],
        "nuc_shape": [c for c in all_non_metadata_cols if "nuc_AreaShape" in c],
        "nuc_texture": [c for c in all_non_metadata_cols if "nuc_Texture" in c],
        "cell_shape": [
            c for c in all_non_metadata_cols if "cell_napari_AreaShape" in c
        ],
        "cell_texture": [
            c for c in all_non_metadata_cols if "cell_napari_Texture" in c
        ],
    }
    feat_cols["morphological"] = (
        feat_cols["nuc_shape"]
        + feat_cols["nuc_texture"]
        + feat_cols["cell_shape"]
        + feat_cols["cell_texture"]
    )

    if prune_loc_feats:
        feat_cols = {
            k: [c for c in v if not any([f"Center_{s}" in c for s in ["X", "Y", "Z"]])]
            for k, v in feat_cols.items()
        }

    return feat_cols


def split_data(df, stratify_on="kg_score", random_state=0, test_size=0.25):
    """splits df and returns dict of inds/dfs for test and train split"""
    inds = np.arange(len(df))
    inds_train, inds_test, _, _ = train_test_split(
        inds,
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_on],
    )
    return {
        "dfs": {
            "train": df.loc[inds_train, :].reset_index(drop=True),
            "test": df.loc[inds_test, :].reset_index(drop=True),
        },
        "indices": {"train": inds_train, "test": inds_test},
    }


def check_low_var_cols(df_train):
    """for each probe pair check that no cols are zero variance"""
    feat_cols = get_feat_col_groups(df_train)
    my_feats = [
        e
        for k, v in feat_cols.items()
        for e in v
        if e not in COLUMN_GROUPS["loc_score"]
    ]

    probe_pairs = get_probe_pairs(df_train)
    for probe_pair in probe_pairs:
        df_pp = subset_to_probe_pair(probe_pair)

        selector = VarianceThreshold()
        selector.fit(df_pp[my_feats])
        assert np.all(selector.get_support())


def drop_xyz_locs(df):
    """xyz positions in field shouldn't matter to single cell stats"""
    cols = [
        c
        for c in df.columns
        if any(x in c for x in ["Center_X", "Center_Y", "Center_Z"])
    ]
    return df.drop(cols, axis="columns")


def drop_zernike_fish_feats(
    df,
    feat_patterns=[
        "seg_probe_561_AreaShape_Zernike",
        "seg_probe_638_AreaShape_Zernike",
    ],
):
    "these are mostly nans"
    cols = [f for f in df.columns if any([x in f for x in feat_patterns])]
    return df.drop(cols, axis="columns")
