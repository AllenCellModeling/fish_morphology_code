#!/usr/bin/env python


import pandas as pd
from pathlib import PurePath, Path
import quilt3
import fire


def distribute_raw(
    pkg_dest="aics/integrated_transcriptomics_structural_organization_hipsc_cm",
    readme="README.md",
    s3_bucket="s3://allencell",
    edit=True,
):

    # either edit package if it exists or create new
    if edit:
        p = quilt3.Package.browse(pkg_dest, registry=s3_bucket)
    else:
        p = quilt3.Package()

    # fetch data csv used to make manuscript figures
    p_manuscript_data = quilt3.Package.browse(
        "aics/integrated_transcriptomics_structural_organization_hipsc_cm",
        "s3://allencell",
    )
    p_manuscript_data["revised_manuscript_plots"]["data.csv"].fetch("./data.csv")

    # load figure data and fish fov metadata df
    img_df = pd.read_csv("./data.csv")
    fish_fov_df = pd.read_csv("fish_fovs.csv")

    # merge fish metadata into img_df
    img_df = img_df.merge(
        fish_fov_df, on=["fov_path", "plate_name", "Cell age"], how="left"
    )

    # replace NaN 488 channel with ACTN2-mEGFP
    img_df["channel488"] = img_df["channel488"].fillna("ACTN2-mEGFP")

    # keep only metadata columns
    keep_cols = [
        "Dataset",
        "napariCell_ObjectNumber",
        "fov_path",
        "Cell age",
        "replate_date",
        "image_date",
        "plate_name",
        "well_position",
        "Type",
        "channel488",
        "channel546",
        "channel647",
        "Cell line",
    ]

    remove_cols = [x for x in img_df.columns if x not in keep_cols]
    img_df = img_df.drop(columns=remove_cols)

    # deduplicate images
    img_df = img_df.drop_duplicates(
        subset=["Dataset", "fov_path", "channel488", "channel546", "channel647"],
        keep="first",
    ).reset_index(drop=True)

    replate_groups = {
        "2019-05-17": 1,
        "2019-05-21": 2,
        "2019-05-24": 3,
        "2019-06-04": 4,
        "2020-06-07": 5,
        "2020-08-28": 6,
        "2020-09-08": 7,
    }

    channel_defs = {0: "638", 1: "nuc", 2: "bf1", 3: "561", 4: "bf2", 5: "488"}

    translate_channel = {
        "638": "channel647",
        "488": "channel488",
        "561": "channel546",
        "nuc": "dapi",
        "bf1": "brightfield",
        "bf2": "brightfield",
    }

    raw_image_row_list = []
    for index, row in img_df.iterrows():

        img_plate = row["plate_name"]
        img_type = row["Type"]
        fov_path = row["fov_path"]

        # get experiment
        experiment = replate_groups[row["replate_date"]]

        # if image is live, remove  C3.tif from end of fov_path
        if img_type == "Live":
            fov_path = fov_path.split("_C3.tif")[0]

        # go through channels
        for i in range(0, 6):
            img_path = PurePath(f"{fov_path}_C{i}.tif")
            img_name = img_path.name

            if Path(img_path).exists():
                p.set(f"raw_images/{img_type}/{img_plate}/{img_name}", img_path)

                # turn row into dictionary to use for populating raw image data frame
                current_channel_dict = row.to_dict()
                current_channel_dict["experiment_number"] = experiment
                current_channel_dict["image_name"] = img_name
                current_channel_dict["xml"] = False

                # get channel definition
                channel_alias = channel_defs[i]

                channel_name = translate_channel[channel_alias]
                if channel_name in ["channel488", "channel546", "channel647"]:
                    channel_name = row[channel_name]
                current_channel_dict["Channel"] = channel_alias
                current_channel_dict["Channel description"] = channel_name

                raw_image_row_list.append(current_channel_dict)

            else:
                print(f"skipping {img_path}")
                continue

            # check if xml file exists
            xml_file = PurePath(f"{fov_path}_C{i}.tif.xml")
            xml_name = xml_file.name
            xml_file_alt = PurePath(f"{fov_path}_C{i}.tif.xml").parent
            xml_file_alt = PurePath(f"{xml_file_alt}/xml_files/{xml_name}")

            if Path(xml_file).exists():
                pass

            elif Path(xml_file_alt).exists():
                xml_name = xml_file_alt.name
                xml_file = xml_file_alt

            else:
                print(f"skipping {xml_name}")
                continue

            p.set(f"raw_images/{img_type}/{img_plate}/{xml_name}", xml_file)

            current_channel_dict = row.to_dict()
            current_channel_dict["experiment_number"] = experiment
            current_channel_dict["image_name"] = xml_name
            current_channel_dict["xml"] = True

            # get channel definition
            channel_alias = channel_defs[i]

            channel_name = translate_channel[channel_alias]
            if channel_name in ["channel488", "channel546", "channel647"]:
                channel_name = row[channel_name]
            current_channel_dict["Channel"] = channel_alias
            current_channel_dict["Channel description"] = channel_name

            raw_image_row_list.append(current_channel_dict)

    # save raw image df (including xml files that exist)
    raw_image_df = pd.DataFrame(raw_image_row_list)
    raw_image_file_manifest = "raw_images_manifest.csv"

    raw_image_df = raw_image_df.drop(columns=["channel647", "channel546", "channel488"])
    raw_image_df.to_csv(raw_image_file_manifest, index=False)

    p.set(f"raw_images/README.md", readme)
    p.set(
        f"raw_images/{raw_image_file_manifest}", PurePath(raw_image_file_manifest).name
    )

    p.push(pkg_dest, s3_bucket, message="raw images")


if __name__ == "__main__":
    fire.Fire(distribute_raw)
