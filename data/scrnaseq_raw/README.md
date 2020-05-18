# AICS cardio scRNA-seq data 

**Overview:** This dataset contains single cell RNA-sequencing data from in vitro hiPSC derived cardiomyocytes
and includes both raw (fastq) and processed data (count matrix).

## Cardiomyocyte differentiation and sample collection 
hiPSCs were differentiated into cardiomyocytes using two protocols: small molecule (Lian et al. 2012) and 
cytokine (Palpant et al. 2015). Samples were collected for sequencing at 4 time points: 1) D0 time point
before the initiation of differentiation, 2) D12/D14 after differentiation was initiated, 3) D24/D26, and D90.

## Library preparation and sequencing
Single cell libraries were prepared using the Split-Seq method. (see manuscript for more details)

## Contents 
- fastq: raw fastq files

- read_assignment: read assignment to gene

- supplementary_files:
    - raw count matrix in mtx format:
        * raw_counts.mtx (count matrix)
        * genes.csv (row names for matrix)
        * cells.csv (column names for matrix)

    - table of cell metadata (day, protocol, cell line, etc):
        * cell_metadata.csv

    - same raw count matrix and cell metadata as above but saved as objects in .RData file
        * scrnaseq_cardio_20191210.RData

    - raw and normalized counts and cell metadata as an anndata object
        - scrnaseq_cardio_20191016.h5ad
