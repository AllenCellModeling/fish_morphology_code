#!/usr/bin/env python


r"""
OME-TIFF image processing
"""


from ome_types import OME, from_xml
from ome_types.model import TiffData

import re
import xml.etree.ElementTree as ET


def clean_metadata(meta: str) -> OME:
    OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    meta = meta.replace("http://www.openmicroscopy.org/Schemas/OME/2011-06", OME_NS)
    meta = meta.replace("Spinning Disk Confocal", "SpinningDiskConfocal")
    meta = meta.replace("Alternate source", "Other")
    meta = meta.replace("DisplayName", "UserName")

    # remove amplificationgain from detectorsettings and add to detector
    omexmlroot = ET.fromstring(meta)
    namespace_matches = re.match(r"\{.*\}", omexmlroot.tag)
    if namespace_matches is not None:
        namespace = namespace_matches.group(0)
    else:
        raise ValueError("XML does not contain a namespace")
    for detector_index, detector in enumerate(
        omexmlroot.findall(f"./{namespace}Instrument/{namespace}Detector")
    ):
        detector_id = detector.get("ID")
        for detectorsettings_index, detectorsettings in enumerate(
            omexmlroot.findall(
                f"./{namespace}Image/{namespace}Pixels/{namespace}Channel/{namespace}DetectorSettings"
            )
        ):
            thedetectorid = detectorsettings.get("ID")
            if detectorsettings.attrib.get("AmplificationGain"):
                # remove attribute from DetectorSettings
                attr = detectorsettings.attrib.pop("AmplificationGain")
                if thedetectorid == detector_id:
                    # add attribute to matching Detector
                    detector.attrib["AmplificationGain"] = attr

    # move Experimenter to before Instrument
    childnodeorder = [
        f"{{{OME_NS}}}Project",
        f"{{{OME_NS}}}Dataset",
        f"{{{OME_NS}}}Folder",
        f"{{{OME_NS}}}Experiment",
        f"{{{OME_NS}}}Plate",
        f"{{{OME_NS}}}Screen",
        f"{{{OME_NS}}}Experimenter",
        f"{{{OME_NS}}}ExperimenterGroup",
        f"{{{OME_NS}}}Instrument",
        f"{{{OME_NS}}}Image",
        f"{{{OME_NS}}}StructuredAnnotations",
        f"{{{OME_NS}}}ROI",
    ]
    omexmlroot[:] = sorted(
        omexmlroot, key=lambda child: childnodeorder.index(child.tag)
    )

    # Register namespace
    ET.register_namespace("", OME_NS)

    # Write out cleaned XML to string
    meta = ET.tostring(omexmlroot, encoding="unicode", method="xml")

    omemeta = from_xml(meta)

    # drop all tiff data blocks in favor one one default in-order tiffdata
    omemeta.images[0].pixels.tiff_data_blocks = [
        TiffData(plane_count=len(omemeta.images[0].pixels.planes))
    ]

    return omemeta
