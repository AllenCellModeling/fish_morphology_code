# Data set for _Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes_

This data package contains the input data for all analyses in the manuscript [_Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes_](https://www.biorxiv.org/content/10.1101/2020.05.26.081083v1) in a compute-friendly form.
Not all of these data were used in the manuscript, but all of the data used in the manuscript are included here.

## Overview
Notably, we provide 478 fields of view containing approximately 5000 segmented single cells in different stages of cardiomyogenesis, imaged in five channels:
- Brightfield
- Hoechst nuclear stain
- Endogenously GFP-tagged alpha-actinin-2 structure
- Two FISH probes per cell (eight probes overall)

Also included are
- expert annotations of these ~5000 segmented cells
- FISH images of cells without a GFP labeled structure (~30 probes)
- scRNA-seq (Split-seq) data collected on approximately 22,000 cells that underwent similar differentiation protocols as the cells we imaged

## Organization
The data in this package is organized into separate data sets, reflecting different data of different types (scRNA-seq vs FISH / image data), and different downstream processing / feature derivation.

The data creation and processing pipeline is organized according to the following schematic:
![Data pipeline schematic](resources/Website_schematic_data_flow_20200310_v2.png)

The data sets included in this package are:

```bash
    cardio_diff_manuscript
    ├── 2d_autocontrasted_fields_and_single_cells
    ├── 2d_autocontrasted_single_cell_features
    ├── 2d_nonstructure_fields
    ├── 2d_nonstructure_single_cell_features
    ├── 2d_nuclear_masks
    ├── 2d_segmented_fields
    ├── 3d_actn2_segmentation
    ├── automated_local_and_global_structure
    ├── manuscript_plots
    ├── probe_localization
    ├── probe_structure_classifier
    ├── scrnaseq_data
    └── scrnaseq_raw
```

Notably absent from this release are the raw 3D images from which our 2D images are derived.
These will be included shortly.

## Access
The data are programmatically accessible via `quilt`, and is also (somewhat) browse-able via this web ui.

### Bulk download
To download the entire data set, install the `quilt` python package using
```bash
pip install quilt
```
and then
```python
import quilt3
b = quilt3.Bucket("s3://allencell")
b.fetch("aics/integrated_transcriptomics_structural_organization_hipsc_cm/", "./")
```

### Download specific files or data sets
To download only certain individual files, navigate the web ui here to the specific file you are interested in, and use the `DOWNLOAD FILE` button in the upper right of the page.

To download specific folders/directories of data, similarly use the web ui to find the directory you want, and check the `<> CODE` tab at the top of the page for the python code that downloads that specific subset of data.

### Programmatic access
To access the data via the python quilt API, install `quilt` via `pip`, and then load the package with:

```python
pkg = quilt3.Package.browse(
    "aics/integrated_transcriptomics_structural_organization_hipsc_cm",
    "s3://allencell",
)
```
Instructions for interacting with quilt packages in Python can be found [here](https://docs.quiltdata.com/walkthrough/getting-data-from-a-package).

## Citation
```
@article {Gerbin2020.05.26.081083,
	author = {Gerbin, Kaytlyn A and Grancharova, Tanya and Donovan-Maiye, Rory and Hendershott, Melissa C and Brown, Jackson and Dinh, Stephanie Q and Gehring, Jamie L and Hirano, Matthew and Johnson, Gregory R and Nath, Aditya and Nelson, Angelique and Roco, Charles M and Rosenberg, Alex B and Sluzewski, M Filip and Viana, Matheus P and Yan, Calysta and Zaunbrecher, Rebecca J and Cordes Metzler, Kimberly R and Menon, Vilas and Palecek, Sean P and Seelig, Georg and Gaudreault, Nathalie and Knijnenburg, Theo and Rafelski, Susanne M and Theriot, Julie A and Gunawardane, Ruwanthi N},
	title = {Cell states beyond transcriptomics: integrating structural organization and gene expression in hiPSC-derived cardiomyocytes},
	elocation-id = {2020.05.26.081083},
	year = {2020},
	doi = {10.1101/2020.05.26.081083},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/05/27/2020.05.26.081083},
	eprint = {https://www.biorxiv.org/content/early/2020/05/27/2020.05.26.081083.full.pdf},
	journal = {bioRxiv}
}
```

## License
For questions on licensing please refer to https://www.allencell.org/terms-of-use.html.

## Contact
Allen Institute for Cell Science E-mail: cells@alleninstitute.org

## Feedback
Feedback on benefits and issues you discovered while using this data package is greatly appreciated. [Feedback Form](https://forms.gle/GUBC3zU5kuA8wyS17)
