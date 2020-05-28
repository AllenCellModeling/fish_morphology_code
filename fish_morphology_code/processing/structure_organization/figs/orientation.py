import os
import lmfit
import argparse
import numpy as np
import pandas as pd
from skimage import io as skio
from skimage import feature as skfeature
from matplotlib import cm, pyplot


def get_cropped_cell(input_img, input_msk=None):

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


def exp_decay(x, a):
    return np.exp(-a * x ** 2)


def analyze_orientation(
    raw,
    mask,
    CellLabel,
    nlevels=8,
    dmax=32,
    nangles=16,
    expdecay=0.5,
    plot=True,
    save_fig=None,
):

    """
        This combination of parameters display the best performance in ranking
        10 cells by CV compared to manual ranking.
    """

    dists = np.linspace(0, dmax, dmax + 1)
    angles = np.linspace(0, np.pi, nangles)

    crop = get_cropped_cell(raw * (mask == CellLabel))
    crop = QuantizeImage(crop, nlevels=nlevels)

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
            lmfit.Model(ExpDecay)
            .fit(curve, x=dists, a=0.005, weights=np.exp(-expdecay * dists))
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
                0.12 * dists,
                curve,
                color=colormap(i / (len(angles) - 1))
                if i != hpeak_angle_pos
                else "black",
                label=name,
                linewidth=1 if i != hpeak_angle_pos else 2,
            )
            ax[0, 1].plot(
                0.12 * dists, curve_fit, "--", color=colormap(i / (len(angles) - 1))
            )
            ax[1, 0].plot(
                0.12 * dists,
                curve_reg,
                color=colormap(i / (len(angles) - 1))
                if i != hpeak_angle_pos
                else "black",
                linewidth=1 if i != hpeak_angle_pos else 2,
            )
            ax[1, 0].axvline(x=0.12 * hpeak_dist_pos, linestyle="--", color="black")
            ax[1, 0].axhline(y=hpeak, linestyle="--", color="black")

        ax[0, 1].set_xlabel("Offset distance (µm)", fontsize=fontsize)
        ax[0, 1].set_ylabel("Haralick correlation", fontsize=fontsize)
        ax[0, 1].legend(bbox_to_anchor=(1.01, 1.05))

        ax[1, 0].set_title(
            f"Peak value: {hpeak:1.3f} at distance: {0.12*hpeak_dist} (µm) for angle {hpeak_angle:1.1f}°"
        )
        ax[1, 0].set_xlabel("Offset distance (µm)", fontsize=fontsize)
        ax[1, 0].set_ylabel(
            "Haralick correlation with\nexponetial decay removed", fontsize=fontsize
        )
        ax[1, 0].set_ylim(0, 1)

        ax[1, 1].plot(0.12 * dists, avg_corr, color="red", label="mean (reg)")
        ax[1, 1].plot(0.12 * dists, cvr_corr, color="blue", label="cv")
        ax[1, 1].set_title(f"Maximum coefficient of variation: {cvr_corr.max():1.3f}")
        ax[1, 1].set_xlabel("Offset distance (µm)", fontsize=fontsize)
        ax[1, 1].set_ylim(0, 1)
        ax[1, 1].legend()

        ax3 = ax[1, 1].twinx()
        ax3.plot(0.12 * dists, std_corr, color="black")
        ax3.set_ylim(0, std_corr.max())
        ax3.set_ylabel("StDev", fontsize=fontsize)

        pyplot.tight_layout()
        if save_fig is None:
            pyplot.show()
        else:
            fig.savefig(save_fig)
            pyplot.close(fig)

    return {
        "MaxCoeffVar": cvr_corr.max(),
        "HPeak": hpeak,
        "PeakDistance": hpeak_dist,
        "PeakAngle": hpeak_angle,
    }


def process_fov(FOVId, df_fov):

    source = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/output/"

    filename = os.path.join(source, f"data_output_fov_{FOVId}_bkgrd.tif")

    data = skio.imread(filename)

    mask = skio.imread(df_fov.MaskFileName.values[0])

    df_or = pd.DataFrame()
    for CellLabel in df_fov.index:
        s = pd.Series(
            AnalyzeOrientation(
                raw=data[0], mask=mask[-1], CellLabel=CellLabel, plot=False
            ),
            name=CellLabel,
        )
        df_or = df_or.append(s)

    #
    # Concatenate CNN predictions, Radon angle map and single cells masks. Save reesult
    #

    data_final = np.vstack([data, mask[-1:]]).astype(np.float32)

    save_name = filename.replace("data_output_", "").replace("_bkgrd", "")

    df_or.to_csv(save_name.replace(".tif", ".orientation"))
    skio.imsave(save_name, data_final)


if __name__ == "__main__":

    #
    # Get FOV id
    #

    parser = argparse.ArgumentParser(
        description="Runs Radon analysis on a particular FOV"
    )
    parser.add_argument("--fov", help="Full path to FOV", required=True)
    args = vars(parser.parse_args())

    #
    # Run FOV
    #

    df_fov = pd.read_csv("database/database.csv", index_col=1)

    df_cell = pd.read_csv("database/cell_database.csv")

    df_cell["FOVId"] = df_fov.loc[df_cell.RawFileName].FOVId.values

    df_cell = df_cell.set_index(["FOVId", "CellId"])

    df_cell = df_cell.sort_index()

    FOVId = int(args["fov"])

    ProcessFOV(FOVId=FOVId, df_fov=df_cell.loc[(FOVId,)])
