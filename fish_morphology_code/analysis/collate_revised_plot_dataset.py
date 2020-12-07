""" Load and collate quilt packages for making revised Cell Systems manuscripts plots"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import quilt3
import fish_morphology_code.analysis.plots
from fish_morphology_code.analysis.notebook_utils import BAR_PLOT_COLUMNS


def collate_revised(
    fish_plates="/allen/aics/gene-editing/FISH/2019/chaos/2020_fish_figures/paper_fish_plates_20201027.csv",
    plot_csv="collated_plot_data.csv",
):

    # get fish plates with replate dates
    df_plates = pd.read_csv(fish_plates, engine="python")
    df_plates["Plate Name"] = df_plates["Plate Name"].astype(int)
    df_plates = df_plates.rename(
        columns={"Plate Name": "plate_name", "Replate Date": "replate_date"}
    )

    # Load FISH data

    # Original fish data for 8 genes
    df_feats_original = fish_morphology_code.analysis.plots.load_main_feat_data(
        username_packagename="tanyasg/2d_autocontrasted_single_cell_features",
        bucket="s3://allencell-internal-quilt",
        feats_csv="features/a749d0e2_cp_features.csv",
        use_cached=False,
        dest_dir=Path("tmp_quilt_data"),
    )

    # Bonus fish data for 8 additional genes
    df_feats_bonus = fish_morphology_code.analysis.plots.load_main_feat_data(
        username_packagename="tanyasg/2d_autocontrasted_single_cell_features2",
        bucket="s3://allencell-internal-quilt",
        feats_csv="features/1493afe2_cp_features.csv",
        use_cached=False,
        dest_dir=Path("tmp_quilt_data_round_2"),
    )

    # ACTN2 probe old fish (one well w/ ACTN2/TTN)
    df_feats_actn2_old = fish_morphology_code.analysis.plots.load_main_feat_data(
        username_packagename="tanyasg/2d_autocontrasted_single_cell_features_actn2",
        bucket="s3://allencell-internal-quilt",
        feats_csv="features/5a192f08_cp_features.csv",
        use_cached=False,
        dest_dir=Path("tmp_quilt_data_actn2"),
    )

    # ACTN2 probe new fish (additional ACTN2/TTN wells)
    df_feats_actn2_new = fish_morphology_code.analysis.plots.load_main_feat_data(
        username_packagename="tanyasg/2d_autocontrasted_single_cell_features_actn2_2",
        bucket="s3://allencell-internal-quilt",
        feats_csv="features/e1c7dd15_cp_features.csv",
        use_cached=False,
        dest_dir=Path("tmp_quilt_data_actn2_2"),
    )

    # Flag gfp negative cells in each FISH data set
    # original data set had 0 scores for cells to indicate absence of structure/gfp
    df_feats_original["gfp_keep"] = (df_feats_original.kg_structure_org_score > 0) & (
        df_feats_original.mh_structure_org_score > 0
    )

    # other fish data were scored afterword; 0 in "no_structure" indicates absence of structure/gfp in cell; all other cells are NaN
    df_feats_bonus["gfp_keep"] = df_feats_bonus.no_structure.isna()
    df_feats_actn2_old["gfp_keep"] = df_feats_actn2_old.no_structure.isna()
    df_feats_actn2_new["gfp_keep"] = df_feats_actn2_new.no_structure.isna()

    # Concatenate all fish features from cell profiler
    all_fish_df = pd.concat(
        [df_feats_original, df_feats_bonus, df_feats_actn2_old, df_feats_actn2_new]
    ).reset_index(drop=True)

    all_fish_df.napariCell_ObjectNumber = all_fish_df.napariCell_ObjectNumber.astype(
        int
    ).astype(str)
    all_fish_df["Type"] = "FISH"

    # add replate date and dataset to fish df
    all_fish_df = all_fish_df.merge(
        df_plates[["plate_name", "replate_date", "Dataset"]], on=["plate_name"]
    )

    all_fish_df["ge_wellID"] = (
        all_fish_df.plate_name.astype(str) + "-" + all_fish_df.well_position.astype(str)
    )

    # Load Classifier organization score packages from quilt

    # Original Fish
    p_gs = quilt3.Package.browse(
        "matheus/assay_dev_fish_analysis", "s3://allencell-internal-quilt"
    )

    df_gs = fish_morphology_code.analysis.plots.fetch_df(
        "metadata.csv",
        p_gs,
        use_cached=False,
        dest_dir=Path("tmp_quilt_scores_original"),
    ).drop("Unnamed: 0", axis="columns")

    # Bonus Fish
    p_gs_bonus = quilt3.Package.browse(
        "tanyasg/struct_scores_bonus", "s3://allencell-internal-quilt"
    )

    df_gs_bonus = fish_morphology_code.analysis.plots.fetch_df(
        "metadata.csv",
        p_gs_bonus,
        use_cached=False,
        dest_dir=Path("tmp_quilt_scores_bonus"),
    ).drop("Unnamed: 0", axis="columns")

    # ACTN2 old Fish
    p_gs_actn2 = quilt3.Package.browse(
        "tanyasg/struct_scores_actn2", "s3://allencell-internal-quilt"
    )

    df_gs_actn2 = fish_morphology_code.analysis.plots.fetch_df(
        "metadata.csv",
        p_gs_actn2,
        use_cached=False,
        dest_dir=Path("tmp_quilt_scores_actn2"),
    ).drop("Unnamed: 0", axis="columns")

    # ACTN2 new Fish
    p_gs_actn2_2 = quilt3.Package.browse(
        "tanyasg/struct_scores_actn2_2", "s3://allencell-internal-quilt"
    )

    df_gs_actn2_2 = fish_morphology_code.analysis.plots.fetch_df(
        "metadata.csv",
        p_gs_actn2_2,
        use_cached=False,
        dest_dir=Path("tmp_quilt_scores_actn2_2"),
    ).drop("Unnamed: 0", axis="columns")

    # Concatenate FISH structure scores
    all_fish_struct_scores_df = pd.concat(
        [df_gs, df_gs_bonus, df_gs_actn2, df_gs_actn2_2]
    ).reset_index(drop=True)

    all_fish_struct_scores_df.napariCell_ObjectNumber = all_fish_struct_scores_df.napariCell_ObjectNumber.astype(
        int
    ).astype(
        str
    )
    all_fish_struct_scores_df = all_fish_struct_scores_df.rename(
        columns={"original_fov_location": "fov_path"}
    )
    all_fish_struct_scores_df = all_fish_struct_scores_df.drop(
        columns=["Age", "Dataset"]
    )

    # Merge structure + cellprofiler features
    fish_df = pd.merge(
        left=all_fish_df,
        right=all_fish_struct_scores_df,
        on=["napariCell_ObjectNumber", "fov_path"],
        how="inner",
    )

    # Clean up columns
    feature_columns_in = [
        "napariCell_AreaShape_Area",
        "napariCell_AreaShape_MinorAxisLength",
        "napariCell_AreaShape_MajorAxisLength",
        "Frac_Area_Background",
        "Frac_Area_DiffuseOthers",
        "Frac_Area_Fibers",
        "Frac_Area_Disorganized_Puncta",
        "Frac_Area_Organized_Puncta",
        "Frac_Area_Organized_ZDisks",
        "Maximum_Coefficient_Variation",
        "Peak_Height",
        "Peak_Distance",
    ]
    metadata_cols_in = [
        "Dataset",
        "napariCell_ObjectNumber",
        "fov_path",
        "cell_age",
        "replate_date",
        "image_date",
        "plate_name",
        "well_position",
        "ge_wellID",
        "mh_structure_org_score",
        "kg_structure_org_score",
        "no_structure",
        "gfp_keep",
        "Type",
        "probe546",
        "probe647",
        "napariCell_Children_seg_probe_561_Count",
        "napariCell_Children_seg_probe_638_Count",
        "IntensitySumIntegrated",
        "IntensitySumIntegratedBkgSub",
    ]

    fish_df = fish_df[metadata_cols_in + feature_columns_in]

    fish_df["Cell aspect ratio"] = (
        fish_df["napariCell_AreaShape_MinorAxisLength"]
        / fish_df["napariCell_AreaShape_MajorAxisLength"]
    )
    fish_df = fish_df.drop(
        columns=[
            "napariCell_AreaShape_MinorAxisLength",
            "napariCell_AreaShape_MajorAxisLength",
        ]
    )

    feature_columns_out = [c for c in fish_df.columns if c not in metadata_cols_in]

    for col in ["probe546", "probe647"]:
        fish_df[col] = fish_df[col].apply(lambda x: x.split("-")[0])

    fish_df = fish_df.rename(
        columns={
            "IntensitySumIntegrated": "alpha-actinin protein intensity (sum)",
            "IntensitySumIntegratedBkgSub": "alpha-actinin protein intensity (sum, background subtracted)",
        }
    )

    n_valid_fishes_in = (~fish_df[["probe546", "probe647"]].isnull()).sum().sum()
    # n_valid_fishes_in

    # Give all probe counts their own columns
    fish_df = fish_morphology_code.analysis.plots.unmelt_probes(fish_df)

    n_valid_fishes_out = (
        (~fish_df[[c for c in fish_df.columns if c.endswith("_count")]].isnull())
        .sum()
        .sum()
    )
    # n_valid_fishes_out

    assert n_valid_fishes_out == n_valid_fishes_in

    # Move to density rather than counts
    count_cols = [c for c in fish_df.columns if c.endswith("count")]

    for count_col in count_cols:
        probe = count_col.split("_")[0]
        density_col = f"{probe}_density"
        fish_df[density_col] = fish_df[count_col] / fish_df.napariCell_AreaShape_Area

    # keep all counts
    # drop all count columns expect the ACTN2
    # drop_count_cols = [c for c in count_cols if c != "ACTN2_count"]
    # fish_df = fish_df.drop(columns=drop_count_cols)

    # Get Live cell data from quilt
    p_gs_live = quilt3.Package.browse(
        "tanyasg/struct_scores_actn2_live", "s3://allencell-internal-quilt"
    )

    df_live = fish_morphology_code.analysis.plots.fetch_df(
        "metadata.csv",
        p_gs_live,
        use_cached=False,
        dest_dir=Path("tmp_quilt_scores_actn2_live"),
    )

    live2fish_rename_dict = {
        "diffuse_fraction_area_covered": "Frac_Area_DiffuseOthers",
        "fibers_fraction_area_covered": "Frac_Area_Fibers",
        "disorganized_puncta_fraction_area_covered": "Frac_Area_Disorganized_Puncta",
        "organized_puncta_fraction_area_covered": "Frac_Area_Organized_Puncta",
        "organized_z_disk_fraction_area_covered": "Frac_Area_Organized_ZDisks",
        "Aspect_Ratio": "Cell aspect ratio",
        "Area": "napariCell_AreaShape_Area",
        "MH_score": "mh_structure_org_score",
        "KG_score": "kg_structure_org_score",
        "Cell age": "cell_age",
        "FOV": "fov_path",
        "cell_id": "napariCell_ObjectNumber",
        "full_cell_total_sum_intensity": "alpha-actinin protein intensity (sum)",
        "full_cell_total_sum_intensity_bg_subtracted": "alpha-actinin protein intensity (sum, background subtracted)",
    }

    df_live = df_live.rename(columns=live2fish_rename_dict)

    # fill nans in feature cols with zeros
    df_live[feature_columns_out] = df_live[feature_columns_out].fillna(0)

    df_live["ge_wellID"] = (
        df_live.plate_name.astype(str) + "-" + df_live.well_position.astype(str)
    )
    df_live["Dataset"] = "Live"
    df_live["Type"] = "Live"

    # Filter live by protein density

    df_live["alpha-actinin protein intensity (density, background subtracted)"] = (
        df_live["alpha-actinin protein intensity (sum, background subtracted)"]
        / df_live["napariCell_AreaShape_Area"]
    )

    # Use 1080 protein density as cutoff threshold (corresponds to 75000 after converting pixel units
    df_live["gfp_keep"] = (
        df_live["alpha-actinin protein intensity (density, background subtracted)"]
        >= 1080
    )

    # Combine FISH + Live features
    df = pd.concat(
        [fish_df, df_live[[c for c in df_live.columns if c in fish_df.columns]]]
    ).reset_index(drop=True)

    # fix units on length / area cols
    pixel_size_xy_in_micrometers = 0.12
    df["napariCell_AreaShape_Area"] = (
        df["napariCell_AreaShape_Area"] * pixel_size_xy_in_micrometers ** 2
    )
    df["Peak_Distance"] = df["Peak_Distance"] * pixel_size_xy_in_micrometers

    # convert transcript density
    density_cols = [c for c in df.columns if c.endswith("_density")]
    for col in density_cols:
        df[col] = df[col] / pixel_size_xy_in_micrometers ** 2

    df = df.rename(
        columns={
            "napariCell_AreaShape_Area": "Cell area (μm^2)",
            "Frac_Area_Background": "Fraction cell area background",
            "Frac_Area_DiffuseOthers": "Fraction cell area diffuse/other",
            "Frac_Area_Fibers": "Fraction cell area fibers",
            "Frac_Area_Disorganized_Puncta": "Fraction cell area disorganized puncta",
            "Frac_Area_Organized_Puncta": "Fraction cell area organized puncta",
            "Frac_Area_Organized_ZDisks": "Fraction cell area organized z-disks",
            "Maximum_Coefficient_Variation": "Max coefficient var",
            "Peak_Height": "Peak height",
            "Peak_Distance": "Peak distance (μm)",
            "cell_age": "Cell age",
            **{col: f"{col.split('_')[0]} (count/μm^2)" for col in density_cols},
        }
    )

    # Calculate mean expert score
    df["Expert structural annotation score (mean)"] = df[
        ["mh_structure_org_score", "kg_structure_org_score"]
    ].mean(axis="columns")

    # standardize cell ages
    df["Cell age"] = df["Cell age"].map(
        {18: 18, 19: 18, 25: 25, 26: 25, 32: 32, 33: 32}
    )

    assert len(df) == len(
        df[["napariCell_ObjectNumber", "fov_path"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Fit regression on old FISH data to predict on all others
    df_old = df[df.Dataset == "OldFish"].copy()

    all_good_scores = (df_old.kg_structure_org_score > 0) & (
        df_old.mh_structure_org_score > 0
    )
    df_old = df_old[all_good_scores]

    feat_cols = [c for c in BAR_PLOT_COLUMNS if c in df.columns]

    df_reg = df_old[feat_cols + ["Expert structural annotation score (mean)"]].copy()

    scaler = StandardScaler()
    scaler.fit(df_reg[feat_cols])

    df_reg[feat_cols] = scaler.transform(df_reg[feat_cols])

    # Predict COS for all cells

    reg = fish_morphology_code.analysis.plots.make_regression(df_reg)
    df["Combined organizational score"] = reg.predict(scaler.transform(df[feat_cols]))

    df.to_csv(plot_csv, index=False)

    return df
