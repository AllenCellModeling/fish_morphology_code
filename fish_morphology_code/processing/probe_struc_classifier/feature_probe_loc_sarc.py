from pathlib import Path
import os

import numpy as np
import pandas as pd
import tifffile

# run this function from the main directory of the fish_morphology_code repo
def compute_probe_loc_feats(
    output_folder="data/probe_sar_loc",
    channel_dict={
        "bf": 0,
        "488": 1,
        "561": 2,
        "405": 3,
        "638": 4,
        "seg_probe_488": 5,
        "seg_probe_561": 6,
        "seg_probe_638": 7,
        "foreback": 8,
        "cell": 9,
    },
    probe_segs={6: "seg_561", 7: "seg_638"},
    plates_date={"20190807": "5500000013", "20190816": "5500000014"},
):

    # all of this should be read in from quilt ordowloaded locally
    seg_folder = Path("quilt_data_contrasted/rescaled_2D_fov_tiff_path")
    class_folder = Path("quilt_data_mattheusv/structure_classifier_features")
    nuc_folder = Path(
        "/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20191122/cp_out_images"
    )

    nuc_csv_path = Path(
        "/allen/aics/gene-editing/FISH/2019/chaos/data/cp_20191122/absolute_metadata.csv"
    )
    sarc_csv_path = Path(
        "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/assay-dev-cardio/AssayDevFishAnalsysis2019-Handoff.csv"
    )
    nuc_csv = pd.read_csv(nuc_csv_path)
    nuc_csv = nuc_csv[
        ["original_fov_location", "rescaled_2D_fov_tiff_path"]
    ].drop_duplicates()
    sarc_csv = pd.read_csv(sarc_csv_path)

    for index, row in sarc_csv.iterrows():
        sarc_csv.loc[index, "fovid"] = row["CellId"].split("-")[1]
    sarc_names = sarc_csv[["ImagePath", "RawPath", "fovid"]]
    sarc_names = sarc_names.drop_duplicates()

    end_string = "_C0.tif"

    output_df = pd.DataFrame()
    count = 0
    for index, row in sarc_names.iterrows():
        img_path = row["RawPath"]
        img_name = img_path.split("/")[-1]
        print("reading " + str(count) + " " + img_name)
        count += 1
        session = img_name.split("_")[2][2]
        well = img_name.split(" [144]")[0][-2:]
        position_endstring = img_name.split("-")[-1][1:]
        position = position_endstring.split(end_string)[0]

        date_of_file = img_name.split("_")[0]
        plate = str(plates_date[date_of_file])

        data_file = (
            str(plate)
            + "_63X_"
            + date_of_file
            + "_S"
            + session
            + "_P"
            + position
            + "_"
            + well
        )
        seg_img_path = os.path.join(
            seg_folder, data_file + "_annotations_corrected_rescaled.ome.tiff"
        )
        class_img_path = os.path.join(
            class_folder,
            row["ImagePath"].split("/")[-1].split("radon")[0] + "bkgrd.tif",
        )

        seg_img = tifffile.imread(seg_img_path)
        class_img = tifffile.imread(class_img_path)

        napari = seg_img[channel_dict["cell"], :, :]

        img_path_tif = img_path.split(end_string)[0]
        nuc_name = (
            nuc_csv.loc[
                nuc_csv["original_fov_location"] == img_path_tif,
                "rescaled_2D_fov_tiff_path",
            ]
            .tolist()[0]
            .split("/")[-1]
            .split(".tiff")[0]
            + "nuc_final_mask.tiff"
        )
        nuc_mask = tifffile.imread(os.path.join(nuc_folder, nuc_name))

        nuc_mask_binary = np.zeros(nuc_mask.shape)
        nuc_mask_binary[nuc_mask == 0] = 1

        cell_masked_nuc = napari * nuc_mask_binary

        for cell in np.unique(cell_masked_nuc):
            if cell > 0:

                data = {
                    "nuc_mask_path": os.path.join(nuc_folder, nuc_name),
                    "RawPath": img_path,
                    "cell_num": cell,
                }

                cell_mask = cell_masked_nuc == cell
                class_mask = class_img[9, :, :] * cell_mask

                cell_px = np.sum(napari == cell)

                nuc = (napari == cell) * nuc_mask
                nuc = nuc.astype(bool)
                nuc_px = np.sum(nuc)
                data.update({"cell_px": cell_px, "nuc_px": nuc_px})

                for probe_channel, seg_id in probe_segs.items():
                    # probe_channel = 6 # TODO: delete this
                    # seg_id = 'seg_561'
                    probe_seg = seg_img[probe_channel, :, :]
                    probe_in_mask = (cell_mask * probe_seg) > 0

                    total_probe_px = np.sum(probe_in_mask)

                    data.update({seg_id + "_total_probe_cyto": total_probe_px})

                    for sarc_class in range(1, 7):
                        probe_px = np.sum(probe_in_mask * (class_mask == sarc_class))
                        class_px = np.sum(class_mask == sarc_class)
                        data.update(
                            {
                                seg_id + "_probe_px_class_" + str(sarc_class): probe_px,
                                seg_id + "_area_px_class_" + str(sarc_class): class_px,
                            }
                        )

                    probe_in_nucleus = nuc * probe_seg
                    probe_in_nucleus = probe_in_nucleus.astype(bool)
                    probe_nuc_px = np.sum(probe_in_nucleus)

                    data.update({seg_id + "_probe_px_nuc": probe_nuc_px})

                output_df = output_df.append(data, ignore_index=True)
                output_df.to_csv(
                    os.path.join(output_folder, "sarc_classification_temp.csv")
                )

    output_df.to_csv(os.path.join(output_folder, "sarc_classification.csv"), index=False)


def merge_final_output_csv():

    output_folder = Path("/allen/aics/microscopy/Calysta/test/fish_struc_seg")
    output_df = pd.read_csv(os.path.join(output_folder, "sarc_classification.csv"))

    output_df["cell_num"] = output_df["cell_num"].astype(int)
    for seg in ["561", "638"]:
        for class_type in range(0, 6):

            # add density inside class
            output_df["seg_" + seg + "_density_class_" + str(class_type)] = (
                output_df["seg_" + seg + "_probe_px_class_" + str(class_type)]
                / output_df["seg_" + seg + "_area_px_class_" + str(class_type)]
            )

            # calculate density outside of class
            probe_px_out = 0
            area_px_out = 0
            for out_class_type in range(0, 6):
                if out_class_type != class_type:
                    probe_px_out += output_df[
                        "seg_" + seg + "_probe_px_class_" + str(out_class_type)
                    ]
                    area_px_out += output_df[
                        "seg_" + seg + "_area_px_class_" + str(out_class_type)
                    ]
            # add density outside of class
            output_df[
                "seg_" + seg + "_probe_px_OUTSIDE_class_" + str(class_type)
            ] = probe_px_out
            output_df[
                "seg_" + seg + "_area_px_OUTSIDE_class_" + str(class_type)
            ] = area_px_out
            output_df["seg_" + seg + "_density_OUTSIDE_class_" + str(class_type)] = (
                probe_px_out / area_px_out
            )

    # Turn nan value to 0, denominator = 0
    for index, row in output_df.iterrows():
        for seg in ["561", "638"]:
            for class_type in range(0, 6):
                if row["seg_" + seg + "_area_px_class_" + str(class_type)] == 0:
                    output_df.loc[
                        index, "seg_" + seg + "_density_class_" + str(class_type)
                    ] = 0

    # Add column of fov path to match with other csvs
    for index, row in output_df.iterrows():
        fov_path = row["RawPath"].split("_C0")[0]
        output_df.loc[index, "fov_path"] = fov_path

    # Clean columns
    for class_type in range(0, 6):
        output_df["area_px_class_" + str(class_type)] = output_df[
            "seg_561_area_px_class_" + str(class_type)
        ]
        for seg in ["561", "638"]:
            output_df.drop(columns=["seg_" + seg + "_area_px_class_" + str(class_type)])

    # Output to csv
    output_df.to_csv(
        Path("/allen/aics/microscopy/Calysta/test/fish_struc_seg/sarc_classification_for_Rory_20200210.csv")
    )
