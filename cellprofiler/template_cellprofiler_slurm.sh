#!/bin/bash

#SBATCH --job-name cellprofiler_test
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 120:00
#SBATCH --mem-per-cpu 10000
#SBATCH --output cp_slurm_test.out

module load anaconda3

source activate cellprofiler-3.1.8

srun template_cellprofiler_run.sh
