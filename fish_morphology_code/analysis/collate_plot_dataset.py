"""Functions for prepping data for plots and plot formatting."""

import numpy as np
import quilt3

from fish_morphology_code.analysis.plots import load_data


# name map for plot-appropriate column names
PRETTY_NAME_MAP = {
    "cell_age": "Cell age",
    "nuclei_count": "Nuclei count",
    "finalnuc_border": "Nucleus touches FOV border",
    "mh_structure_org_score": "Expert structural annotation score (annotator 1)",
    "kg_structure_org_score": "Expert structural annotation score (annotator 2)",
    "napariCell_ObjectNumber": "Cell number",
    "fov_path": "FOV path",
    "rescaled_2D_single_cell_tiff_path": "Rescaled 2D single cell tiff path",
    "structure_org_score": "Expert structural annotation score (mean)",
    "cell_area": "Cell area",
    "cell_area_micrometers_squared": "Cell area (μm^2)",
    "cell_aspect_ratio": "Cell aspect ratio",
    "frac_area_background": "Fraction cell area background",
    "frac_area_messy": "Fraction cell area diffuse/other",
    "frac_area_threads": "Fraction cell area fibers",
    "frac_area_random": "Fraction cell area disorganized puncta",
    "frac_area_regular_dots": "Fraction cell area organized puncta",
    "frac_area_regular_stripes": "Fraction cell area organized z-disks",
    "max_coeff_var": "Max coefficient var",
    "h_peak": "Peak height",
    "peak_distance": "Peak distance",
    "peak_distance_micrometers": "Peak distance (μm)",
    "peak_angle": "Peak angle",
    "IntensityMedian": "Alpha-actinin intensity (median)",
    "IntensityMedianNoUnits": "Alpha-actinin intensity (median, normalized per day)",
    "IntensityIntegrated": "Alpha-actinin intensity (integrated)",
    "HPRT1_count": "HPRT1 count",
    "COL2A1_count": "COL2A1 count",
    "H19_count": "H19 count",
    "ATP2A2_count": "ATP2A2 count",
    "MYH6_count": "MYH6 count",
    "MYH7_count": "MYH7 count",
    "BAG3_count": "BAG3 count",
    "TCAP_count": "TCAP count",
    "consensus_structure_org_score_roundup": "Expert structural annotation score (round-up)",
    "consensus_structure_org_score_grouped": "Expert structural annotation score (grouped)",
    "structure_org_weighted_linear_model_all": "Combined organizational score",
    "structure_org_weighted_linear_model_all_rounded": "Combined organizational score (rounded)",
    "HPRT1_density": "HPRT1 density",
    "COL2A1_density": "COL2A1 density",
    "H19_density": "H19 density",
    "ATP2A2_density": "ATP2A2 density",
    "MYH6_density": "MYH6 density",
    "MYH7_density": "MYH7 density",
    "BAG3_density": "BAG3 density",
    "TCAP_density": "TCAP density",
    "HPRT1_count_per_micrometer_squared": "HPRT1 (count/μm^2)",
    "COL2A1_count_per_micrometer_squared": "COL2A1 (count/μm^2)",
    "H19_count_per_micrometer_squared": "H19 (count/μm^2)",
    "ATP2A2_count_per_micrometer_squared": "ATP2A2 (count/μm^2)",
    "MYH6_count_per_micrometer_squared": "MYH6 (count/μm^2)",
    "MYH7_count_per_micrometer_squared": "MYH7 (count/μm^2)",
    "BAG3_count_per_micrometer_squared": "BAG3 (count/μm^2)",
    "TCAP_count_per_micrometer_squared": "TCAP (count/μm^2)",
    "FISH_probe": "FISH probe",
    "FISH_probe_count": "FISH probe count",
    "FISH_probe_density": "FISH probe density",
    "FISH_probe_density_count_per_micrometer_squared": "FISH probe (count/μm^2)",
    "MYH7-MYH6_normalized_difference": "MYH7-MYH6 relative transcript abundance (normalized)",
    "MYH6_localization": "MYH6 localization",
    "MYH7_localization": "MYH7 localization",
    "MYH6_probe_frac_regular_dots_gain_over_random": "Enrichment of MYH6 transcript localization to organized puncta",
    "MYH7_probe_frac_regular_dots_gain_over_random": "Enrichment of MYH7 transcript localization to organized puncta",
    "MYH6_probe_frac_regular_stripes_gain_over_random": "Enrichment of MYH6 transcript localization to organized z-disks",
    "MYH7_probe_frac_regular_stripes_gain_over_random": "Enrichment of MYH7 transcript localization to organized z-disks",
    "MYH6_probe_frac_threads_gain_over_random": "Enrichment of MYH6 transcript localization to fibers",
    "MYH7_probe_frac_threads_gain_over_random": "Enrichment of MYH7 transcript localization to fibers",
    "MYH6_probe_frac_threads_regular_dots_regular_stripes_gain_over_random": "Enrichment of MYH6 transcript localization to fibers, organized puncta, and organized z-disks",
    "MYH7_probe_frac_threads_regular_dots_regular_stripes_gain_over_random": "Enrichment of MYH7 transcript localization to fibers, organized puncta, and organized z-disks",
    "MYH6_dist_to_alpha-actinin_segmentation": "MYH6 probe distance to alpha-actinin segmentation (mean)",
    "MYH7_dist_to_alpha-actinin_segmentation": "MYH7 probe distance to alpha-actinin segmentation (mean)",
}


def replace_cnn_class_map(string, dicitonary):
    for k, v in dicitonary.items():
        string = string.replace(k, v)
    return string


def collate_plot_dataset(pixel_size_xy_in_micrometers=0.12):

    # load main feature data
    df, _, _ = load_data()

    # load datasets
    p_feats = quilt3.Package.browse(
        "tanyasg/2d_autocontrasted_single_cell_features",
        "s3://allencell-internal-quilt",
    )
    df_feats = p_feats["features"]["a749d0e2_cp_features.csv"]()
    df_probe_channel_key = df_feats[
        [
            "napariCell_ObjectNumber",
            "fov_path",
            "probe546",
            "probe647",
            "probe_561_loc_score",
            "probe_638_loc_score",
        ]
    ]

    df_probe_channel_key = df_probe_channel_key.rename(
        columns={"probe546": "probe_561", "probe647": "probe_638"}
    )
    for col in ["probe_561", "probe_638"]:
        df_probe_channel_key[col] = df_probe_channel_key[col].apply(
            lambda x: x.split("-")[0]
        )

    p_probe_metrics = quilt3.Package.browse(
        "calystay/probe_localization", "s3://allencell-internal-quilt"
    )
    df_probe_metrics = p_probe_metrics["metadata.csv"]()
    df_probe_metrics = df_probe_metrics.rename(
        columns={"original_fov_location": "fov_path"}
    )
    df_probe_metrics = df_probe_metrics[
        [
            "fov_path",
            "napariCell_ObjectNumber",
            "seg_561_cell_abs_dist_struc_total_mean",
            "seg_638_cell_abs_dist_struc_total_mean",
        ]
    ]

    p_pl = quilt3.Package.browse(
        "calystay/probe_structure_classifier", "s3://allencell-internal-quilt"
    )
    df_pl = p_pl["metadata.csv"]().rename(columns={"original_fov_location": "fov_path"})
    category_map = {
        "class_0": "background",
        "class_1": "messy",
        "class_2": "threads",
        "class_3": "random_dots",
        "class_4": "regular_dots",
        "class_5": "regular_stripes",
    }
    df_pl.columns = [replace_cnn_class_map(col, category_map) for col in df_pl.columns]

    # merge datasets
    df_myh67 = (
        df.merge(df_probe_channel_key)
        .merge(df_probe_metrics)
        .dropna(subset=["MYH6_density", "MYH7_density"])
        .merge(df_pl, how="inner", on=["fov_path", "napariCell_ObjectNumber"])
        .reset_index(drop=True)
    )

    # compute probe localization enrichments
    for feature in ["threads", "regular_dots", "regular_stripes"]:
        df_myh67[f"seg_561_probe_frac_{feature}"] = df_myh67[
            f"seg_561_probe_px_{feature}"
        ] / (df_myh67["seg_561_total_probe_cyto"] + df_myh67["seg_561_probe_px_nuc"])
        df_myh67[f"seg_638_probe_frac_{feature}"] = df_myh67[
            f"seg_638_probe_px_{feature}"
        ] / (df_myh67["seg_638_total_probe_cyto"] + df_myh67["seg_638_probe_px_nuc"])

        df_myh67[f"seg_561_probe_frac_{feature}_gain_over_random"] = (
            df_myh67[f"seg_561_probe_frac_{feature}"] - df_myh67[f"frac_area_{feature}"]
        )
        df_myh67[f"seg_638_probe_frac_{feature}_gain_over_random"] = (
            df_myh67[f"seg_638_probe_frac_{feature}"] - df_myh67[f"frac_area_{feature}"]
        )

    # enrichments for sum of threads, reg dots, and reg stripes
    df_myh67["seg_561_probe_frac_threads_regular_dots_regular_stripes"] = (
        df_myh67["seg_561_probe_px_threads"]
        + df_myh67["seg_561_probe_px_regular_dots"]
        + df_myh67["seg_561_probe_px_regular_stripes"]
    ) / (df_myh67["seg_561_total_probe_cyto"] + df_myh67["seg_561_probe_px_nuc"])
    df_myh67["seg_638_probe_frac_threads_regular_dots_regular_stripes"] = (
        df_myh67["seg_638_probe_px_threads"]
        + df_myh67["seg_638_probe_px_regular_dots"]
        + df_myh67["seg_638_probe_px_regular_stripes"]
    ) / (df_myh67["seg_638_total_probe_cyto"] + df_myh67["seg_638_probe_px_nuc"])
    df_myh67[
        "seg_561_probe_frac_threads_regular_dots_regular_stripes_gain_over_random"
    ] = df_myh67["seg_561_probe_frac_threads_regular_dots_regular_stripes"] - (
        df_myh67["frac_area_threads"]
        + df_myh67["frac_area_regular_dots"]
        + df_myh67["frac_area_regular_stripes"]
    )
    df_myh67[
        "seg_638_probe_frac_threads_regular_dots_regular_stripes_gain_over_random"
    ] = df_myh67["seg_638_probe_frac_threads_regular_dots_regular_stripes"] - (
        df_myh67["frac_area_threads"]
        + df_myh67["frac_area_regular_dots"]
        + df_myh67["frac_area_regular_stripes"]
    )
    name_map = {"seg_638": "MYH7", "seg_561": "MYH6"}
    df_myh67.columns = [
        replace_cnn_class_map(col, name_map) for col in df_myh67.columns
    ]
    df_myh67 = df_myh67.drop(columns=[c for c in df_myh67.columns if "OUTSIDE" in c])

    # add normalized difference metric to data frame ~ (MYH7-MYH6)/(MYH7+MYH6)
    df_myh67["MYH7-MYH6_normalized_difference"] = (
        df_myh67["MYH7_density"] / df["MYH7_density"].dropna().median()
        - df_myh67["MYH6_density"] / df["MYH6_density"].dropna().median()
    ) / (
        df_myh67["MYH7_density"] / df["MYH7_density"].dropna().median()
        + df_myh67["MYH6_density"] / df["MYH6_density"].dropna().median()
    )

    # probe loc annotations -> grouped semantic category names
    sarc_symb_map = {
        "n": "Low expression",
        "0": "Non-sarcomeric",
        "1": "Sarcomeric",
        "2": "Sarcomeric",
        "3": "Sarcomeric",
        np.nan: np.nan,
    }
    assert (
        set(df_myh67["probe_638_loc_score"].unique())
        .union(set(df_myh67["probe_561_loc_score"].unique()))
        .issubset(sarc_symb_map.keys())
    )
    df_myh67["probe_638_loc_score"] = df_myh67["probe_638_loc_score"].map(sarc_symb_map)
    df_myh67["probe_561_loc_score"] = df_myh67["probe_561_loc_score"].map(sarc_symb_map)
    df_myh67 = df_myh67.rename(
        columns={
            "probe_561_loc_score": "MYH6_localization",
            "probe_638_loc_score": "MYH7_localization",
            "MYH6_cell_abs_dist_struc_total_mean": "MYH6_dist_to_alpha-actinin_segmentation",
            "MYH7_cell_abs_dist_struc_total_mean": "MYH7_dist_to_alpha-actinin_segmentation",
        }
    )[
        [
            "napariCell_ObjectNumber",
            "fov_path",
            "MYH7-MYH6_normalized_difference",
            "MYH6_localization",
            "MYH7_localization",
            "MYH6_probe_frac_regular_dots_gain_over_random",
            "MYH6_probe_frac_regular_stripes_gain_over_random",
            "MYH6_probe_frac_threads_gain_over_random",
            "MYH6_probe_frac_threads_regular_dots_regular_stripes_gain_over_random",
            "MYH7_probe_frac_regular_dots_gain_over_random",
            "MYH7_probe_frac_regular_stripes_gain_over_random",
            "MYH7_probe_frac_threads_gain_over_random",
            "MYH7_probe_frac_threads_regular_dots_regular_stripes_gain_over_random",
            "MYH6_dist_to_alpha-actinin_segmentation",
            "MYH7_dist_to_alpha-actinin_segmentation",
        ]
    ]

    # convert area and density cols to real units
    density_cols_orig_no_units = [
        "HPRT1_density",
        "COL2A1_density",
        "H19_density",
        "ATP2A2_density",
        "MYH6_density",
        "MYH7_density",
        "BAG3_density",
        "TCAP_density",
    ]
    for col in density_cols_orig_no_units:
        df[f"{col}_per_micrometer_squared".replace("density", "count")] = (
            df[col] / (pixel_size_xy_in_micrometers) ** 2
        )
    df["cell_area_micrometers_squared"] = (
        df["cell_area"] * pixel_size_xy_in_micrometers ** 2
    )

    # get peak distance too
    df["peak_distance_micrometers"] = df["peak_distance"] * pixel_size_xy_in_micrometers

    # normalize alpha-actinin intensity independently per day
    df["IntensityMedianNoUnits"] = 0
    for i, cell_age in enumerate(sorted(df["cell_age"].unique())):
        condition = df["cell_age"] == cell_age
        data = df[condition]["IntensityMedian"].copy()
        df.loc[condition, "IntensityMedianNoUnits"] = data / data.median()

    # round linear model score to nearest in and clamp to target domain
    df["structure_org_weighted_linear_model_all_rounded"] = np.rint(
        df["structure_org_weighted_linear_model_all"]
    )
    df["structure_org_weighted_linear_model_all_rounded"] = np.clip(
        df["structure_org_weighted_linear_model_all_rounded"], 1, 5
    ).astype(int)

    # merge in MYH6 & MYH7 specific stuff
    df = df.merge(df_myh67, how="left")

    # drop columns not used in plots
    dropcols_all = [
        "IntensityMedianBkgSub",
        "IntensityIntegratedBkgSub",
        "cell_area",
        "peak_distance",
        "HPRT1_density",
        "COL2A1_density",
        "H19_density",
        "ATP2A2_density",
        "MYH6_density",
        "MYH7_density",
        "BAG3_density",
        "TCAP_density",
    ]
    df = df.drop(columns=dropcols_all)

    # rename cols in dfs
    df = df.rename(columns=PRETTY_NAME_MAP)

    # enforce int dtype in some cols
    int_cols = [
        "Nuclei count",
        "Expert structural annotation score (annotator 1)",
        "Expert structural annotation score (annotator 2)",
        "Cell number",
        "Expert structural annotation score (round-up)",
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)

    # done
    return df
