
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from fish_morphology_code.analysis.data_ingestion import (
    get_probe_pairs,
    subset_to_probe_pair,
)


def remove_low_var_feat_cols(adata, threshold=0.0):
    """
    checks  anndata for low/zero var feature cols and removes them.
    each check is done independently for each probe pair, cols with low var in any subset are removed from all.
    """

    drop_cols = set()
    probe_pairs = get_probe_pairs(adata)
    for probe_pair in probe_pairs:
        adata_pp = subset_to_probe_pair(adata, probe_pair)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(adata_pp.X)
        drop_cols |= set(adata_pp.var.index[~selector.get_support()])
    return adata[:, [c for c in adata.var.index if c not in drop_cols]].copy()


def z_score_feats(adata, probe_subsets=True):
    """z-score feat data, save params un adata.uns, scaled data in adata.layers["z-scored"].
       if probe_subsets=True, z-score the probe features individually for each probe."""

    scaler = StandardScaler().fit(adata.X)
    if probe_subsets:
        probes = adata.obs["FISH_probe"].unique()
        scalers_probes = {}
        for probe in probes:
            cell_inds = adata.obs["FISH_probe"] == probe
            feat_inds = adata.var["channel"] == "FISH"
            x = adata[cell_inds, :][:, feat_inds].X
            scalers_probes[probe] = StandardScaler().fit(x)

    adata_out = adata.copy()
    adata_out.layers["z-scored"] = scaler.transform(adata.X)
    adata_out.uns["z-score params"] = {
        "all": {"means": scaler.mean_, "scales": scaler.scale_}
    }
    if probe_subsets:
        for probe in probes:
            cell_inds = adata.obs["FISH_probe"] == probe
            feat_inds = adata.var["channel"] == "FISH"
            x = adata[cell_inds, :][:, feat_inds].X.copy()
            new_x = scalers_probes[probe].transform(x)
            feats_int = [i for i, f in enumerate(feat_inds) if f]
            cells_int = [i for i, c in enumerate(cell_inds) if c]
            for i, c in enumerate(cells_int):
                for j, f in enumerate(feats_int):
                    adata_out.layers["z-scored"][c, f] = new_x[i, j]
            adata_out.uns["z-score params"][probe] = {
                "means": scalers_probes[probe].mean_,
                "scales": scalers_probes[probe].scale_,
            }
    return adata_out


def bootstrap(data=np.random.standard_normal(100), stat=np.median, N=1000):
    """
    data is a 1D numpy array, default = np.random.standard_normal(100)
    stat is a scalar statistic that can be computed on that array, default=np.median
    N is the number of bootstrap replicates, default=1000
    """

    boot_stats = np.array(
        [stat(np.random.choice(data, size=data.shape, replace=True)) for i in range(N)]
    )

    return boot_stats


def conditional_bootstrap(
    adata,
    feature="GNXP density cell",
    probe_col="FISH_probe",
    score_col="kg_structure_org_score",
    age_col="cell_age",
    stat=np.median,
    N_boots=1000,
):
    """bootrap estimates of stat on adata.obs[feature], conditioned on probe_col, score_col, and age_col"""

    df_subsets_median = (
        adata.obs[[probe_col, score_col, age_col]]
        .drop_duplicates()
        .sort_values(by=[probe_col, age_col, score_col])
        .reset_index(drop=True)
    )

    boot_data_median_by_day = {}
    for i, row in df_subsets_median.iterrows():
        (p, s, a) = row[probe_col], row[score_col], row[age_col]
        condition = (
            (adata.obs[probe_col] == p).values
            & (adata.obs[score_col] == s).values
            & (adata.obs[age_col] == a).values
        )
        data = adata.obs[condition][feature].values
        boot_data_median_by_day[(p, s, a)] = bootstrap(data=data, stat=stat, N=N_boots)

    df_median = pd.concat(
        [
            pd.DataFrame(
                {
                    "Probe": p,
                    "Structure score": s,
                    "Cell age": a,
                    "Median FISH density": d,
                }
            )
            for (p, s, a), d in boot_data_median_by_day.items()
        ]
    ).reset_index(drop=True)

    return df_median
