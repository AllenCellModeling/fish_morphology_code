
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from anndata import AnnData


def _determine_region(feat):
    if feat.startswith("finalnuc_"):
        return "nuclear"
    elif feat.startswith("napariCell_"):
        return "cell"
    else:
        return None


def _determine_channel(feat):
    if "_structure_" in feat:
        return "structure"
    elif "_bf_" in feat:
        return "bright field"
    elif "_nuc_" in feat:
        return "DNA"
    elif "_probe_" in feat:
        return "FISH"
    elif "" in feat:
        return "segmentation"
    else:
        return None


def _determine_type(feat):
    if "_Center_" in feat:
        return "location"
    elif "_Count" in feat:
        return "count"
    elif "_Texture_" in feat:
        return "texture"
    elif "_AreaShape_" in feat:
        return "shape"
    else:
        return None


def _determine_probe_wavelength(feat):
    if "probe_638" in feat:
        return "638"
    elif "probe_561" in feat:
        return "561"
    else:
        return None


def make_anndata_feats(
    df_feats,
    obs_cols=[
        "ImageNumber",
        "ImagePath",
        "ImageFailed",
        "napariCell_ObjectNumber",
        "rescaled_2D_fov_tiff_path",
        "rescaled_2D_single_cell_tiff_path",
        "fov_path",
        "well_position",
        "microscope",
        "image_date",
        "probe488",
        "probe546",
        "probe647",
        "plate_name",
        "cell_line",
        "cell_age",
        "napariCell_nuclei_Count",
        "finalnuc_border",
        "finalnuc_ObjectNumber",
        "finalnuc_Parent_FilterNuc",
        "nuccentroid_ObjectNumber",
        "mh_structure_org_score",
        "kg_structure_org_score",
    ],
):
    """make feature df into anndata: X=feat values, obs=metadata, var=feature annotations"""

    # dice into main feature data (X) and metadata (obs)
    X_cols = [
        c
        for c in df_feats.columns
        if (
            any(c.startswith(x) for x in ["napariCell_", "finalnuc_"])
            and c not in obs_cols
        )
    ]
    obs = df_feats[obs_cols]
    X = df_feats[X_cols]
    assert len(df_feats.columns) == len(obs_cols + X_cols)

    # feature annotations to use for subsetting
    var = pd.DataFrame({"name": X.columns})
    var["region"] = var["name"].apply(_determine_region)
    var["channel"] = var["name"].apply(_determine_channel)
    var["type"] = var["name"].apply(_determine_type)
    var["probe_wavelength"] = var["name"].apply(_determine_probe_wavelength)
    assert var["region"].isnull().sum() == 0
    assert var["channel"].isnull().sum() == 0
    assert var["type"].isnull().sum() == 0
    assert (
        len(var) - var["probe_wavelength"].isnull().sum()
        == (var["channel"] == "FISH").sum()
    )

    return AnnData(X=X, obs=obs, var=var.set_index("name"))


_OBS_PROBE_MAP = {"488": "probe488", "561": "probe546", "638": "probe647"}


def tidy_by_probe(adata, probe_wavelengths=["561", "638"]):
    """remove probe pairings and treat each probe as individual obs on duplicataed other feats."""

    adata_s = {p: "" for p in probe_wavelengths}

    for pw in probe_wavelengths:
        feats = (adata.var["probe_wavelength"] == pw) | (
            adata.var["probe_wavelength"].isnull()
        )
        adata_s[pw] = adata[:, feats]
        adata_s[pw].var.index = adata_s[pw].var.index.str.replace(f"{pw}_", "")

        adata_s[pw].var = adata_s[pw].var.drop("probe_wavelength", axis="columns")
        adata_s[pw].obs = (
            adata_s[pw]
            .obs.drop([v for k, v in _OBS_PROBE_MAP.items() if k != pw], axis="columns")
            .rename({_OBS_PROBE_MAP[pw]: "FISH_probe"}, axis="columns")
        )

    adata_c = adata_s[probe_wavelengths[0]]
    for pw in probe_wavelengths[1:]:
        adata_c = adata_c.concatenate(adata_s[pw])
    adata_c.uns = adata.uns.copy()

    return adata_c


def prune_nans(adata_in, thresh=0.1, axis="columns"):
    """drop row/col if frac of nans in row/col is greater than thresh"""
    adata = adata_in.copy()
    if axis == "columns":
        adata = adata[:, np.isnan(adata.X).mean(axis=0) < thresh]
    elif axis == "rows":
        adata = adata[np.isnan(adata.X).mean(axis=1) < thresh, :]
    else:
        return ValueError
    return adata


def iteratively_prune(adata, threshs=[0.1, 0.01]):
    """iterate between dropping rows/cols on sucessively more stringent thresholds"""
    adata_out = adata.copy()
    for thresh in threshs:
        for axis in ["columns", "rows"]:
            adata_out = prune_nans(adata_out, thresh=thresh, axis=axis)
    adata_out.obs = adata_out.obs.reset_index(drop=True)
    return adata_out


def drop_bad_struct_scores(adata, exclude_scores=[-1, 0]):
    """-1 is a bad cell and 0 was non expressing i think?"""
    adata_out = adata[
        (
            ~adata.obs[["mh_structure_org_score", "kg_structure_org_score"]].isin(
                [0, -1]
            )
        ).all(axis="columns"),
        :,
    ].copy()
    adata_out.obs = adata_out.obs.reset_index(drop=True)
    return adata_out


def drop_xyz_locs(adata):
    """xyz positions in field shouldn't matter to single cell stats"""
    return adata[:, adata.var["type"] != "location"].copy()


def remap_struct_scores(
    adata,
    struct_score_cols=["mh_structure_org_score", "kg_structure_org_score"],
    map_dict={1: 1, 2: 1, 3: 2, 4: 3, 5: 3},
):
    """remap structure scores from 1:5 to 1:3 or whatever"""
    adata_out = adata.copy()
    for col in struct_score_cols:
        adata_out.obs[col] = adata_out.obs[col].map(map_dict)
    return adata_out


def split_data(
    adata, stratify_on="kg_structure_org_score", random_state=0, test_size=0.25
):
    """splits anndata and returns dict of inds/anndatas for test and train split"""
    inds = np.arange(len(adata))

    indices = {"train": [], "test": []}
    indices["train"], indices["test"], _, _ = train_test_split(
        inds,
        adata,
        test_size=test_size,
        random_state=random_state,
        stratify=adata.obs[stratify_on],
    )

    adata = {split: adata[inds, :] for split, inds in indices.items()}
    for split, ad in adata.items():
        adata[split].obs = adata[split].obs.reset_index(drop=True)
        adata[split].uns["split"] = split
        adata[split].uns["indices"] = indices[split]

    return adata


def get_probe_pairs(adata, probe_name_cols=["probe546", "probe647"]):
    """unique probe pairs in dataset"""
    return list(list(x) for x in adata.obs[probe_name_cols].drop_duplicates().values)


def subset_to_probe_pair(adata, probe_pair=["MYH6", "MYH7"]):
    """subset the anndata to only a certain probe pair"""
    adata_out = adata[
        np.all(adata.obs[["probe546", "probe647"]] == probe_pair, axis=1), :
    ].copy()
    adata_out.obs = adata_out.obs.reset_index(drop=True)
    return adata_out


def remove_low_var_feat_cols(adatas, threshold=0.0):
    """
    checks train anndata for low/zero var feature cols and removes them from both test and train anndatas.
    each check is done independently for each probe pair, cols with low var in any subset are removed from all.
    """

    drop_cols = set()
    probe_pairs = get_probe_pairs(adatas["train"])
    for probe_pair in probe_pairs:
        adata_pp = subset_to_probe_pair(adatas["train"], probe_pair)

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(adata_pp.X)
        drop_cols |= set(adata_pp.var.index[~selector.get_support()])
    return {
        k: v[:, [c for c in v.var.index if c not in drop_cols]].copy()
        for k, v in adatas.items()
    }


def z_score_feats(adatas, probe_subsets=True):
    """z-score train and test feat data based on train params, return params as well as scaled anndata.
       if probe_subsets=True, z-score the probe features individually for each probe."""

    scaler = StandardScaler().fit(adatas["train"].X)
    if probe_subsets:
        probes = adatas["train"].obs["FISH_probe"].unique()
        scalers_probes = {}
        ad = adatas["train"]
        for probe in probes:
            cell_inds = ad.obs["FISH_probe"] == probe
            feat_inds = ad.var["channel"] == "FISH"
            x = ad[cell_inds, :][:, feat_inds].X
            scalers_probes[probe] = StandardScaler().fit(x)

    adatas_out = {split: ad.copy() for split, ad in adatas.items()}
    for split, ad in adatas_out.items():
        adatas_out[split].layers["z-scored"] = scaler.transform(ad.X)
        adatas_out[split].uns["z-score params"] = {
            "all": {"means": scaler.mean_, "scales": scaler.scale_}
        }
        if probe_subsets:
            for probe in probes:
                cell_inds = ad.obs["FISH_probe"] == probe
                feat_inds = ad.var["channel"] == "FISH"
                x = ad[cell_inds, :][:, feat_inds].X.copy()
                new_x = scalers_probes[probe].transform(x)
                feats_int = [i for i, f in enumerate(feat_inds) if f]
                cells_int = [i for i, c in enumerate(cell_inds) if c]
                for i, c in enumerate(cells_int):
                    for j, f in enumerate(feats_int):
                        adatas_out[split].layers["z-scored"][c, f] = new_x[i, j]
                adatas_out[split].uns["z-score params"][probe] = {
                    "means": scalers_probes[probe].mean_,
                    "scales": scalers_probes[probe].scale_,
                }
    return adatas_out


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


def make_contingency_table(
    adata, groupby=["FISH_probe", "cell_age", "kg_structure_org_score"]
):
    tab = adata.obs.groupby(groupby).count()
    assert (tab.min(axis="columns") == tab.max(axis="columns")).all()
    tab = tab.loc[:, ["ImageNumber"]].rename({"ImageNumber": "Count"}, axis="columns")
    return tab.reset_index()
