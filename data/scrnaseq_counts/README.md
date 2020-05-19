# AICS cardio scRNA-seq data 

**Overview:** Pakcage contains count matrix from single cell RNA-sequencing of in vitro hiPSC derived cardiomyocytes

## Cardiomyocyte differentiation and sample collection 
hiPSCs were differentiated into cardiomyocytes using two protocols: small molecule (Lian et al. 2012) and 
cytokine (Palpant et al. 2015). Samples were collected for sequencing at 4 time points: 1) D0 time point
before the initiation of differentiation, 2) D12/D14 after differentiation was initiated, 3) D24/D26, and D90.

## Library preparation and sequencing
Single cell libraries were prepared using the Split-Seq method. (see manuscript for more details)

## Contents 
- `counts`:
    - `raw_counts.mtx`: raw count matrix in mtx format; gene x cell
    - `genes.csv`: row names for matrix
    - `cells.csv`: column names for matrix

- `anndata`:
    - `scrnaseq_cardio_20191016.h5ad`: anndata format object with normalized count matrix (cell x gene); normalized = log1p(counts normalized by cell * 1e4)

- `supplementary_files`:
    - `scrnaseq_cardio_20191210.RData`: same raw count matrix and cell metadata as in `counts/` but saved as objects in .RData file

