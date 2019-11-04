#!/usr/bin/env python


import os
import pandas as pd


def image_set_list(quilt_path, index, local_path, **kwargs):

    """
    Create image set data frame in format accepted by cellprofiler

    Args:
        quilt_path (str): location of image (relative to quilt package)
        index (int): image index
        local_path (str): path to local directory where images in quilt package are stored

    Returns:
        image_set_df (pd.DataFrame): image set in format accepted by cellprofiler
    """

    image_path = "file:" + os.path.join(local_path, quilt_path)
    dir_name = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    new_frame = pd.DataFrame(
        {
            "Group_Number": [kwargs["Group_Number"]],
            "Group_Index": index,
            "URL_bf": image_path,
            "URL_fore_back": image_path,
            "URL_nuc": image_path,
            "URL_probe_561": image_path,
            "URL_probe_638": image_path,
            "URL_seg_561": image_path,
            "URL_seg_638": image_path,
            "URL_structure": image_path,
            "PathName_bf": dir_name,
            "PathName_fore_back": dir_name,
            "PathName_nuc": dir_name,
            "PathName_probe_561": dir_name,
            "PathName_probe_638": dir_name,
            "PathName_seg_561": dir_name,
            "PathName_seg_638": dir_name,
            "PathName_structure": dir_name,
            "FileName_bf": file_name,
            "FileName_fore_back": file_name,
            "FileName_nuc": file_name,
            "FileName_probe_561": file_name,
            "FileName_probe_638": file_name,
            "FileName_seg_561": file_name,
            "FileName_seg_638": file_name,
            "FileName_structure": file_name,
            "Series_bf": [kwargs["Series_bf"]],
            "Series_fore_back": [kwargs["Series_fore_back"]],
            "Series_nuc": [kwargs["Series_nuc"]],
            "Series_probe_561": [kwargs["Series_probe_561"]],
            "Series_probe_638": [kwargs["Series_probe_638"]],
            "Series_seg_561": [kwargs["Series_seg_561"]],
            "Series_seg_638": [kwargs["Series_seg_638"]],
            "Series_structure": [kwargs["Series_structure"]],
            "Frame_bf": [kwargs["Frame_bf"]],
            "Frame_fore_back": [kwargs["Frame_fore_back"]],
            "Frame_nuc": [kwargs["Frame_nuc"]],
            "Frame_probe_561": [kwargs["Frame_probe_561"]],
            "Frame_probe_638": [kwargs["Frame_probe_638"]],
            "Frame_seg_561": [kwargs["Frame_seg_561"]],
            "Frame_seg_638": [kwargs["Frame_seg_638"]],
            "Frame_structure": [kwargs["Frame_structure"]],
            "Channel_bf": [kwargs["Channel_bf"]],
            "Channel_fore_back": [kwargs["Channel_fore_back"]],
            "Channel_nuc": [kwargs["Channel_nuc"]],
            "Channel_probe_561": [kwargs["Channel_probe_561"]],
            "Channel_probe_638": [kwargs["Channel_probe_638"]],
            "Channel_seg_561": [kwargs["Channel_seg_561"]],
            "Channel_seg_638": [kwargs["Channel_seg_638"]],
            "Channel_structure": [kwargs["Channel_structure"]],
            "ObjectsURL_napari_cell": image_path,
            "ObjectsPathName_napari_cell": dir_name,
            "ObjectsFileName_napari_cell": file_name,
            "ObjectsSeries_napari_cell": [kwargs["ObjectsSeries_napari_cell"]],
            "ObjectsFrame_napari_cell": [kwargs["ObjectsFrame_napari_cell"]],
            "ObjectsChannel_napari_cell": [kwargs["ObjectsChannel_napari_cell"]],
            "Metadata_C": [kwargs["Metadata_C"]],
            "Metadata_ChannelName": [kwargs["Metadata_ChannelName"]],
            "Metadata_ColorFormat": [kwargs["Metadata_ColorFormat"]],
            "Metadata_FileLocation": [kwargs["Metadata_FileLocation"]],
            "Metadata_Frame": [kwargs["Metadata_Frame"]],
            "Metadata_Plate": [kwargs["Metadata_Plate"]],
            "Metadata_Series": [kwargs["Metadata_Series"]],
            "Metadata_Site": [kwargs["Metadata_Site"]],
            "Metadata_SizeC": [kwargs["Metadata_SizeC"]],
            "Metadata_SizeT": [kwargs["Metadata_SizeT"]],
            "Metadata_SizeX": [kwargs["Metadata_SizeX"]],
            "Metadata_SizeY": [kwargs["Metadata_SizeY"]],
            "Metadata_SizeZ": [kwargs["Metadata_SizeZ"]],
            "Metadata_T": [kwargs["Metadata_T"]],
            "Metadata_Well": [kwargs["Metadata_Well"]],
            "Metadata_Z": [kwargs["Metadata_Z"]],
        }
    )

    return new_frame
