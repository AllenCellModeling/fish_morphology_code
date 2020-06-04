import os
import lmfit
import argparse
import numpy as np
import pandas as pd
from quilt3 import Package
from skimage import io as skio
from skimage import feature as skfeature
from matplotlib import cm, pyplot


def get_cropped_cell(input_img, input_msk=None):

    """
    Extracts a cell out a FOV based on the segmentation mask
    """

    if input_msk is None:
        [y, x] = np.nonzero(input_img)
    else:
        [y, x] = np.nonzero(input_msk)

    print(f"Effective Area: {x.size}")

    bins = np.percentile(input_img[y, x], [2, 98])

    print(f"Bins for percentile normalization: {bins}")

    crop = input_img[y.min() : y.max(), x.min() : x.max()]

    crop[crop > 0] = np.clip(crop[crop > 0], *bins)

    return crop.astype(np.uint16)


def quantize_image(input_img, nlevels=8):

    """
    Quantize the pixels intensity in nlevels for orientation
    calculation.
    """

    vmax = input_img.max()

    bins = (
        [0]
        + np.percentile(input_img[input_img > 0], np.linspace(0, 100, nlevels + 1))[
            :-1
        ].tolist()
        + [1 + vmax]
    )

    input_dig = np.digitize(input_img, bins) - 1

    print(f"Frequency of each level: {np.bincount(input_dig.flatten())}")

    return input_dig


def exp_decay_func(x, a):
    return np.exp(-a * x ** 2)


def analyze_orientation(
    raw,
    mask,
    CellLabel,
    nlevels=8,
    dmax=32,
    nangles=16,
    decay_value=0.5,
    plot=True,
    save_fig=None,
):

    """
    This combination of default parameters display the best performance in ranking
    10 cells by CV compared to manual ranking.
    """

    dists = np.linspace(0, dmax, dmax + 1)
    angles = np.linspace(0, np.pi, nangles)

    crop = get_cropped_cell(raw * (mask == CellLabel))
    crop = quantize_image(crop, nlevels=nlevels)

    glcm = skfeature.texture.greycomatrix(
        image=crop, distances=dists, angles=angles, levels=nlevels + 1
    )
    corr = skfeature.texture.greycoprops(glcm[1:, 1:], prop="correlation").T
    corr = np.abs(corr)

    eps = 1e-1  # Regularizer for the mean to avoid too low values
    std_corr = corr.std(axis=0)
    avg_corr = eps + (1 - eps) * corr.mean(axis=0)
    cvr_corr = std_corr / avg_corr

    corr_fit = np.zeros_like(corr)
    corr_reg = np.zeros_like(corr)

    for i, curve in enumerate(corr):
        z = (
            lmfit.Model(exp_decay_func)
            .fit(curve, x=dists, a=0.005, weights=np.exp(-decay_value * dists))
            .best_fit
        )
        corr_fit[i] = z
        corr_reg[i] = curve - z

    hpeak = corr_reg.max()
    hpeak_dist_pos = corr_reg.max(axis=0).argmax()
    hpeak_dist = dists[hpeak_dist_pos]
    hpeak_angle_pos = corr_reg.max(axis=1).argmax()
    hpeak_angle = 180 * angles[hpeak_angle_pos] / np.pi

    if plot:

        # Whether or not to show the result
        fontsize = 16

        colormap = cm.get_cmap("cool", len(angles))

        print(f"Cell: {CellLabel}")
        print(f"Max coefficient of variation: {cvr_corr.max():1.3f}")
        print(
            f"Highest peak: {hpeak:1.3f} at distance {hpeak_dist} and angle {hpeak_angle:1.1f}°"
        )

        fig, ax = pyplot.subplots(2, 2, figsize=(12, 8))

        ax[0, 0].imshow(crop, cmap="gray")
        ax[0, 0].axis("off")

        for i, (curve, curve_fit, curve_reg) in enumerate(
            zip(corr, corr_fit, corr_reg)
        ):

            name = f"{180*angles[i]/np.pi:1.0f}°"

            ax[0, 1].plot(
                dists,
                curve,
                color=colormap(i / (len(angles) - 1))
                if i != hpeak_angle_pos
                else "black",
                label=name,
                linewidth=1 if i != hpeak_angle_pos else 2,
            )
            ax[0, 1].plot(dists, curve_fit, "--", color=colormap(i / (len(angles) - 1)))
            ax[1, 0].plot(
                dists,
                curve_reg,
                color=colormap(i / (len(angles) - 1))
                if i != hpeak_angle_pos
                else "black",
                linewidth=1 if i != hpeak_angle_pos else 2,
            )
            ax[1, 0].axvline(x=hpeak_dist_pos, linestyle="--", color="black")
            ax[1, 0].axhline(y=hpeak, linestyle="--", color="black")

        ax[0, 1].set_xlabel("Distance (pixels)", fontsize=fontsize)
        ax[0, 1].set_ylabel("Correlation", fontsize=fontsize)
        ax[0, 1].legend(bbox_to_anchor=(1.01, 1.05))

        ax[1, 0].set_title(
            f"Peak value: {hpeak:1.3f} at distance: {hpeak_dist} (pixels) for angle {hpeak_angle:1.1f}°"
        )
        ax[1, 0].set_xlabel("Distance (pixels)", fontsize=fontsize)
        ax[1, 0].set_ylabel("Correlation - exponetial fitting", fontsize=fontsize)

        ax[1, 1].plot(dists, avg_corr, color="red", label="mean (reg)")
        ax[1, 1].plot(dists, cvr_corr, color="blue", label="cv")
        ax[1, 1].set_title(f"Maximum coefficient of variation: {cvr_corr.max():1.3f}")
        ax[1, 1].set_xlabel("Distance (pixels)", fontsize=fontsize)
        ax[1, 1].set_ylim(0, 1)
        ax[1, 1].legend()

        ax3 = ax[1, 1].twinx()
        ax3.plot(dists, std_corr, color="black")
        ax3.set_ylim(0, std_corr.max())
        ax3.set_ylabel("StDev", fontsize=fontsize)

        pyplot.tight_layout()
        if save_fig is None:
            pyplot.show()
        else:
            fig.savefig(save_fig)
            pyplot.close(fig)

    return {
        "Maximum_Coefficient_Variation": cvr_corr.max(),
        "Peak_Height": hpeak,
        "Peak_Distance": hpeak_dist,
        "Peak_Angle": hpeak_angle,
    }


def process_fov(FOVId, df_fov):

    filename = f"../output/fov_{FOVId}.tif"

    data = skio.imread(filename)

    df_or = pd.DataFrame()
    for CellLabel in df_fov.index:
        s = pd.Series(
            analyze_orientation(
                raw=data[0], mask=data[-1], CellLabel=CellLabel, plot=False
            ),
            name=CellLabel,
        )
        df_or = df_or.append(s)

    df_or.index = df_or.index.rename('CellId')
    df_or.to_csv(filename.replace(".tif", ".csv"))


if __name__ == "__main__":

    # Get FOV id
    parser = argparse.ArgumentParser(
        description="Runs Radon analysis on a particular FOV"
    )
    parser.add_argument("--fov", help="Full path to FOV", required=True)
    args = vars(parser.parse_args())
    FOVId = int(args["fov"])

    # Downlaod the datasets from Quilt if there is no local copy
    ds_folder = "../database/"

    if not os.path.exists(os.path.join(ds_folder, "metadata.csv")):

        pkg = Package.browse(
            "matheus/assay_dev_datasets", "s3://allencell-internal-quilt"
        ).fetch(ds_folder)

    metadata = pd.read_csv(os.path.join(ds_folder, "metadata.csv"))

    df_fov = pd.read_csv(
        os.path.join(ds_folder, metadata.database_path[0]), index_col=1
    )

    df_cell = pd.read_csv(os.path.join(ds_folder, metadata.cell_database_path[0]))

    # Merge dataframes

    df_cell["FOVId"] = df_fov.loc[df_cell.RawFileName].FOVId.values

    df_cell = df_cell.set_index(["FOVId", "CellId"])

    df_cell = df_cell.sort_index()

    # Run FOV
    process_fov(FOVId=FOVId, df_fov=df_cell.loc[(FOVId,)])
