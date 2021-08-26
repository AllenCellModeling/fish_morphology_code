#!/usr/bin/env python


import fire
import pandas as pd
from pathlib import Path

from aicsimageio import AICSImage
from aicsimageio.readers import TiffReader
from aicsimageio.writers import OmeTiffWriter

from fish_morphology_code.processing.ometiff_processing.ometiff_utils import (
    clean_metadata,
)


def run(img_csv="",):
    """
    Merge individual 3i channels in tiff format to create multi-channel ome-tiff image
    Args:
        img_csv (str): csv filename with list of tiffs to be converted into multi-channel ome-tiffs
    """

    img_df = pd.read_csv(img_csv)
    for index, row in img_df.iterrows():
        filepath = row["image_path"]
        ometiff_path = row["ometiff_path"]

        # create out dir if it doesn't exist
        out_dir = Path(ometiff_path).parent
        if out_dir.exists():
            pass
        else:
            out_dir.mkdir(parents=True)

        print(ometiff_path)

        img = AICSImage(filepath, reader=TiffReader)

        # clean metadata
        meta = img.metadata
        omemeta = clean_metadata(meta)

        writer = OmeTiffWriter()
        writer.save(img.data, ometiff_path, ome_xml=omemeta)


def main():
    fire.Fire(run)
