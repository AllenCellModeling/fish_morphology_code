"""Functions for prepping data for plots and plot formatting."""

from pathlib import Path

import numpy as np
import pandas as pd
import quilt3
import logging

from fish_morphology_code.analysis.data_ingestion import (
    make_anndata_feats,
    iteratively_prune,
    drop_bad_struct_scores,
    drop_xyz_locs,
    widen_df,
    tidy_df,
)

from fish_morphology_code.analysis.basic_stats import remove_low_var_feat_cols
from fish_morphology_code.analysis.structure_prediction import (
    prep_human_score_regression_data,
    regress_human_scores_on_feats,
)


def make_small_dataset(
    adata,
    feats_X=[
        "napariCell_AreaShape_Area",
        "napariCell_AreaShape_MajorAxisLength",
        "napariCell_AreaShape_MinorAxisLength",
        "napariCell_Children_seg_probe_561_Count",
        "napariCell_Children_seg_probe_638_Count",
    ],
    feats_obs=[
        "cell_age",
        "napariCell_nuclei_Count",
        "finalnuc_border",
        "mh_structure_org_score",
        "kg_structure_org_score",
        "napariCell_ObjectNumber",
        "fov_path",
        "probe546",
        "probe647",
    ],
    loc_obs=["rescaled_2D_single_cell_tiff_path"],
):
    """Prune giant feature table down to a handful of basic features for simple analysis."""
    # create aspect ratio feature
    adata_X_df = adata[:, feats_X].to_df()
    adata_X_df["napariCell_AreaShape_AspectRatio"] = (
        adata_X_df["napariCell_AreaShape_MinorAxisLength"]
        / adata_X_df["napariCell_AreaShape_MajorAxisLength"]
    )
    adata_X_df = adata_X_df.drop(
        [
            "napariCell_AreaShape_MajorAxisLength",
            "napariCell_AreaShape_MinorAxisLength",
        ],
        axis="columns",
    ).reset_index()

    adata_obs_df = (
        adata.obs[feats_obs + loc_obs]
        .reset_index()
        .rename(columns={"probe546": "probe_561", "probe647": "probe_638"})
    )
    adata_obs_df["consensus_structure_org_score"] = (
        adata_obs_df["kg_structure_org_score"] + adata_obs_df["mh_structure_org_score"]
    ) / 2
    adata_df = adata_obs_df.merge(adata_X_df, how="inner").drop(
        ["index"], axis="columns"
    )

    return adata_df


def pretty_chart(
    chart, font_color="black", title_font_size=16, label_font_size=16, opacity=0.75
):
    """Easier basic formatting for altair charts."""
    config_dict = dict(
        titleColor=font_color,
        labelColor=font_color,
        titleFontSize=title_font_size,
        labelFontSize=label_font_size,
    )

    return (
        chart.configure_header(**config_dict)
        .configure_legend(**config_dict)
        .configure_axisX(**config_dict)
        .configure_axisY(**config_dict)
        .configure_circle(opacity=0.75)
    )


rename_dict = {
    "napariCell_nuclei_Count": "nuclei_count",
    "consensus_structure_org_score": "structure_org_score",
    "napariCell_AreaShape_Area": "cell_area",
    "napariCell_AreaShape_AspectRatio": "cell_aspect_ratio",
    "FracAreaBackground": "frac_area_background",
    "FracAreaMessy": "frac_area_messy",
    "FracAreaThreads": "frac_area_threads",
    "FracAreaRandom": "frac_area_random",
    "FracAreaRegularDots": "frac_area_regular_dots",
    "FracAreaRegularStripes": "frac_area_regular_stripes",
    "SarcomereWidth": "sarcomere_width",
    "SarcomereLength": "sarcomere_length",
    "NRegularStripesVoronoi": "n_regular_stripes_voronoi",
    "RadonDominantAngleEntropyThreads": "radon_entropy_threads",
    "RadonResponseRatioAvgThreads": "radon_response_threads",
    "RadonDominantAngleEntropyRegStripes": "radon_entropy_stripes",
    "RadonResponseRatioAvgRegStripes": "radon_response_stripes",
    "MaxCoeffVar": "max_coeff_var",
    "HPeak": "h_peak",
    "PeakDistance": "peak_distance",
    "PeakAngle": "peak_angle",
}


def load_main_feat_data(rename_dict=rename_dict, use_cached=False):
    # load main feature data
    p_feats = quilt3.Package.browse(
        "tanyasg/2d_autocontrasted_single_cell_features",
        "s3://allencell-internal-quilt",
    )
    df_feats = p_feats["features/a749d0e2_cp_features.csv"]()
    return df_feats


def adata_manipulations(df_feats, rename_dict=rename_dict):
    anndata_logger = logging.getLogger("anndata")
    anndata_logger.setLevel(logging.CRITICAL)
    adata = make_anndata_feats(df_feats)

    # clean up adata
    adata = iteratively_prune(adata)
    adata = drop_bad_struct_scores(adata)
    adata = drop_xyz_locs(adata)

    # check to make sure we have all the cells/feautes we expect
    assert np.isnan(adata.X).sum() == 0
    assert adata.X.shape == (4785, 1065)

    # drop uninformative features
    adata = remove_low_var_feat_cols(adata)
    return adata


def get_global_structure(rename_dict=rename_dict, use_cached=False):
    p_gs = quilt3.Package.browse(
        "matheus/assay_dev_fish_analysis", "s3://allencell-internal-quilt"
    )
    df_gs = p_gs["metadata.csv"]()
    df_gs = df_gs[
        [
            "napariCell_ObjectNumber",
            "original_fov_location",
            "FracAreaBackground",
            "FracAreaMessy",
            "FracAreaThreads",
            "FracAreaRandom",
            "FracAreaRegularDots",
            "FracAreaRegularStripes",
            "MaxCoeffVar",
            "HPeak",
            "PeakDistance",
            "PeakAngle",
            "IntensityMedian",
            "IntensityIntegrated",
            "IntensityMedianBkgSub",
            "IntensityIntegratedBkgSub",
        ]
    ].rename(rename_dict, axis="columns")

    # clean up some columns to use as ids and merge into main dataframe
    df_gs = df_gs.rename(columns={"original_fov_location": "fov_path"})
    df_gs["fov_path"] = df_gs["fov_path"].apply(lambda p: str(Path(p)))
    return df_gs


def group_human_scores(df, rename_dict=rename_dict):
    df["consensus_structure_org_score_roundup"] = np.ceil(
        df["consensus_structure_org_score"]
    ).astype(int)
    df["consensus_structure_org_score_grouped"] = df[
        "consensus_structure_org_score_roundup"
    ].map({1: "1-2", 2: "1-2", 3: "3", 4: "4-5", 5: "4-5"})
    return df


def make_regression_df(rename_dict=rename_dict):
    pass


def make_and_clean_tidy_df(rename_dict=rename_dict):
    pass


def add_densities(rename_dict=rename_dict):
    pass


def load_data():
    """Monster function for loading and munging data for plots."""

    # load main feature data
    df_feats = load_main_feat_data()

    # make anndata from feature data
    # anndata as intermediate because our general purpose cleaning functions are written for anndata rather than pandas objects
    adata = adata_manipulations(df_feats)

    # make a df version of the feature data that only has a handful of simple features
    df_small = make_small_dataset(adata)

    # load in global structure features (DNN area classifier + radon transform stuff)
    df_gs = get_global_structure()

    # merge in the global structure metrics
    df_small = df_small.merge(df_gs)

    # make wide version
    df = widen_df(df_small)

    # group manual human structure scores into coarser bins
    df = group_human_scores(df)

    # aggregate metric for total fraction of cell covered by "regular" ACTN2 structure
    df["frac_area_regular_sum"] = (
        df["frac_area_regular_dots"] + df["frac_area_regular_stripes"]
    )

    # clean up feauter/column names on the dataframes
    df = df.rename(rename_dict, axis="columns")

    # add linear model structure scores
    X_cols = [
        "cell_area",
        "cell_aspect_ratio",
        "frac_area_background",
        "frac_area_messy",
        "frac_area_threads",
        "frac_area_random",
        "frac_area_regular_dots",
        "frac_area_regular_stripes",
        "max_coeff_var",
        "h_peak",
        "peak_distance",
    ]
    y_col = "structure_org_score"
    weight_col = "structure_org_score"
    my_reg_df = prep_human_score_regression_data(df)
    regression = regress_human_scores_on_feats(
        my_reg_df, X_cols=X_cols, y_col=y_col, weight_col=weight_col
    )
    df["structure_org_weighted_linear_model_all"] = regression.predict(
        my_reg_df[X_cols]
    )

    regression_df = pd.DataFrame({"feature": X_cols, "coef": regression.coef_})

    # create version of feature data where FISH probes are unpaired (makes facet plots easier)
    df_tidy = tidy_df(df)

    # clean up feature/column names on the dataframes
    df = df.rename(rename_dict, axis="columns")
    df_tidy = df_tidy.rename(rename_dict, axis="columns")

    # clean up FISH probe names to drop amplifier ID
    df = df.rename(
        columns={
            c: c.replace("-B1", "").replace("-B3", "").replace("-B4", "")
            for c in df.columns
        }
    )
    df_tidy = df_tidy.rename(
        columns={
            c: c.replace("-B1", "").replace("-B3", "").replace("-B4", "")
            for c in df.columns
        }
    )
    df_tidy.FISH_probe = df_tidy.FISH_probe.map(
        {c: c.split("-")[0] for c in df_tidy.FISH_probe.unique()}
    )

    # move from counts to count densities (normalize to cell area)
    count_cols = [c for c in df.columns if ("_count" in c) & ("nuclei" not in c)]
    for col in count_cols:
        df[col.replace("count", "density")] = df[col] / df["cell_area"]
    df_tidy["FISH_probe_density"] = df_tidy["FISH_probe_count"] / df_tidy["cell_area"]

    return df, df_tidy, regression_df
