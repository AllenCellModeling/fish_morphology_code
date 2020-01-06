"""Functions for prepping data for plots a dn plot formatting."""


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
    loc_obs=["rescaled_2D_fov_tiff_path", "rescaled_2D_single_cell_tiff_path"],
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
