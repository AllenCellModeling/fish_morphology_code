import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from fish_morphology_code.analysis.structure_prediction import (
    prep_human_score_regression_data,
    regress_human_scores_on_feats,
)

# matplotlib config
plt.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.sans-serif"] = "arial"
matplotlib.rcParams["font.family"] = "sans-serif"


# color config
DAY_18_COLOR = "#0098EA"
DAY_32_COLOR = "#FFB2FF"
DAY_COLORS = [DAY_18_COLOR, DAY_32_COLOR]
DAY_COLOR_PALETTE = sns.color_palette(DAY_COLORS)
DAY_COLOR_PALETTE_THREE = sns.color_palette(DAY_COLORS + ["grey"])


# columns / facet definitons and order
BAR_PLOT_COLUMNS = [
    "Cell area (μm^2)",
    "Cell aspect ratio",
    "Fraction cell area background",
    "Fraction cell area diffuse/other",
    "Fraction cell area fibers",
    "Fraction cell area disorganized puncta",
    "Fraction cell area organized puncta",
    "Fraction cell area organized z-disks",
    "Max coefficient var",
    "Peak height",
    "Peak distance (μm)",
]

SHORT_FEAT_NAME_MAP = {
    c: c.replace("Fraction cell area", "")
    .replace("(μm^2)", "")
    .replace("(μm)", "")
    .strip()
    .capitalize()
    for c in BAR_PLOT_COLUMNS
}

BAR_PLOT_COLUMNS_SHORT = [SHORT_FEAT_NAME_MAP[v] for v in BAR_PLOT_COLUMNS]

PROBE_ORDER = [
    "MYH6",
    "MYH7",
    "COL2A1",
    "H19",
    "TCAP",
    "ATP2A2",
    "BAG3",
    "HPRT1",
    "BMPER",
    "CNTN5",
    "MEF2C",
    "MYL7",
    "NKX2",
    "PLN",
    "PRSS35",
    "VCAN",
]


def safe(string, sub="_"):
    """sub any non alphanumeric or - or _ character with an underscore or whatever and then lowecase"""
    return (
        re.sub(r"[^\w\-_\. ]", sub, string).replace(" ", "_").replace("__", "_").lower()
    )


def get_regression_coef(
    df,
    X_cols=BAR_PLOT_COLUMNS,
    y_col="Combined organizational score",
    weight_col="Combined organizational score",
):
    """perform a sklearn regression and grab the coeficients"""
    my_reg_df = prep_human_score_regression_data(
        df, all_feats=X_cols, targ_feats=[y_col]
    )
    regression = regress_human_scores_on_feats(
        my_reg_df, X_cols=X_cols, y_col=y_col, weight_col=weight_col
    )
    return regression.coef_


def boot_regression(
    df,
    N=10,
    X_cols=BAR_PLOT_COLUMNS,
    y_col="Combined organizational score",
    weight_col="Combined organizational score",
):
    """bootstrap over get_regression_coef"""
    boot_stats = np.array(
        [
            get_regression_coef(
                df.sample(frac=1, replace=True).reset_index(drop=True),
                X_cols=X_cols,
                y_col=y_col,
                weight_col=weight_col,
            )
            for i in range(N)
        ]
    )
    return pd.DataFrame(boot_stats, columns=X_cols)


# functions for making confidence intervals from the bootstrapped regressions
CI_EXTENT = 0.95

FEATURE_TYPE_MAP = {
    "Cell area (μm^2)": "Cell features",
    "Cell aspect ratio": "Cell features",
    "Fraction cell area background": "Local organization",
    "Fraction cell area diffuse/other": "Local organization",
    "Fraction cell area fibers": "Local organization",
    "Fraction cell area disorganized puncta": "Local organization",
    "Fraction cell area organized puncta": "Local organization",
    "Fraction cell area organized z-disks": "Local organization",
    "Max coefficient var": "Global alignment",
    "Peak height": "Global alignment",
    "Peak distance (μm)": "Global alignment",
}


def ci_low(x, ci_extent=CI_EXTENT):
    return x.quantile((1 - ci_extent) / 2)


def ci_high(x, ci_extent=CI_EXTENT):
    return x.quantile((1 + ci_extent) / 2)


def make_reg_plot_ci_df(
    df_boot_reg_in, ci_extent=0.95, feature_type_map=FEATURE_TYPE_MAP
):
    df_boot_reg = df_boot_reg_in.melt(var_name="Feature", value_name="Feature weight")

    f = {"Feature weight": ["mean", ci_low, ci_high]}
    df_boot_reg_ci = df_boot_reg.groupby(["Feature"]).agg(f).reset_index()
    df_boot_reg_ci.columns = [
        " ".join(x).strip()
        for x in list(
            zip(*[list(df_boot_reg_ci.columns.get_level_values(i)) for i in (0, 1)])
        )
    ]
    df_boot_reg_ci["Feature Type"] = df_boot_reg_ci["Feature"].map(feature_type_map)
    df_boot_reg_ci = df_boot_reg_ci.rename(
        columns={
            "Feature weight mean": "Feature weight (mean)",
            "Feature weight ci_low": "Feature weight (CI low)",
            "Feature weight ci_high": "Feature weight (CI high)",
        }
    )

    return df_boot_reg_ci


def make_regression_bar_plot(
    df_in,
    dims=(4, 3),
    x="Feature",
    order=BAR_PLOT_COLUMNS_SHORT,
    y="Feature weight (mean)",
    hue="Feature Type",
    palette=sns.color_palette("husl", 3),
    ecolor="k",
    elinewidth=1,
    capsize=4,
    ylabel="Feature weight in linear model\n(regression coefficient)",
    title="",
):
    """standardized bar plots of coefficients + error bars from bootstrapped regression"""
    df = df_in.copy()
    df["Feature"] = df["Feature"].map(SHORT_FEAT_NAME_MAP)

    fig, ax = plt.subplots(figsize=dims)
    feat_chart = sns.barplot(
        data=df, x=x, order=order, y=y, hue=hue, palette=palette, dodge=False
    )

    err = np.array(
        (
            (df["Feature weight (mean)"] - df["Feature weight (CI low)"]).values,
            (df["Feature weight (CI high)"] - df["Feature weight (mean)"]).values,
        )
    )

    plt.errorbar(
        x=df.set_index("Feature").loc[order, :].reset_index()["Feature"],
        y=df.set_index("Feature").loc[order, :].reset_index()["Feature weight (mean)"],
        fmt="none",
        yerr=err,
        ecolor=ecolor,
        elinewidth=elinewidth,
        capsize=capsize,
    )

    ax.set_title(title.replace("_", " "))

    sns.despine()

    feat_chart.set_xticklabels(
        feat_chart.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    feat_chart.set(ylabel=ylabel)
    plt.legend(bbox_to_anchor=(1.0, 0.7), frameon=False)

    return fig, ax


def get_pred_true(
    df,
    X_cols=BAR_PLOT_COLUMNS,
    y_col="Combined organizational score",
    weight_col="Combined organizational score",
    meta_cols=["Cell age"],
):
    """get predictions and ground truths from a regression"""
    my_reg_df = prep_human_score_regression_data(
        df, all_feats=X_cols, targ_feats=[y_col] + meta_cols
    )

    regression = regress_human_scores_on_feats(
        my_reg_df, X_cols=X_cols, y_col=y_col, weight_col=weight_col
    )
    df_out = my_reg_df[[y_col] + meta_cols].copy()
    df_out[f"{y_col} predicted"] = regression.predict(my_reg_df[X_cols])
    return df_out


def make_regression_scatter_plot(
    df,
    target,
    dims=(4, 4),
    hue="Cell age",
    hue_order=[18, 32],
    palette=DAY_COLOR_PALETTE,
    title="",
):
    """standardized scatterplot for regressions , comparing ground truth to predictions"""
    dims = (4, 4)
    fig, ax = plt.subplots(figsize=dims)
    ax = sns.scatterplot(
        data=df.sample(frac=1, replace=False).reset_index(drop=True),
        x=target,
        y=f"{target} predicted",
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        linewidth=0,
        alpha=0.5,
        s=10,
    )
    ax.set(
        xlabel=target.replace("_", " "),
        ylabel=f"{target} predicted".replace("_", " "),
        title=title,
    )
    sns.despine()
    plt.legend(bbox_to_anchor=(1.4, 0.6), frameon=False)

    return fig, ax


def boot_spearmanr(
    df,
    x="Combined organizational score",
    y="Alpha-actinin intensity (median, normalized per day)",
    stratify="Cell age",
    N=10,
):
    """bootstrap spearman correlations and construct CIs"""
    df_spr_boot = pd.DataFrame(columns=[stratify, f"Spearman R ({x}, {y})"])
    for condition in df[stratify].dropna().unique():
        df_cond = df[df[stratify] == condition][[x, y]].dropna()
        boot_stats = np.array(
            [
                spearmanr(df_cond.sample(frac=1, replace=True).reset_index(drop=True))[
                    0
                ]
                for i in range(N)
            ]
        )
        df_spr_boot_cond = pd.DataFrame(
            {stratify: [condition] * N, f"Spearman R ({x}, {y})": boot_stats}
        )
        df_spr_boot = df_spr_boot.append(df_spr_boot_cond).reset_index(drop=True)

    f = {f"Spearman R ({x}, {y})": ["mean", ci_low, ci_high]}
    df_spr_boot_corr = df_spr_boot.groupby([stratify]).agg(f).reset_index()
    df_spr_boot_corr.columns = [
        " ".join(x).strip()
        for x in list(
            zip(*[list(df_spr_boot_corr.columns.get_level_values(i)) for i in (0, 1)])
        )
    ]

    df_spr_boot_corr = df_spr_boot_corr.rename(
        columns={
            f"Spearman R ({x}, {y}) mean": f"Spearman R ({x}, {y}) (mean)",
            f"Spearman R ({x}, {y}) ci_low": f"Spearman R ({x}, {y}) (CI low)",
            f"Spearman R ({x}, {y}) ci_high": f"Spearman R ({x}, {y}) (CI high)",
        }
    )

    df_spr_boot_corr[stratify] = df_spr_boot_corr[stratify].astype(str)

    return df_spr_boot_corr
