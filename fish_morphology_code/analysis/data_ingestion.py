import itertools

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
