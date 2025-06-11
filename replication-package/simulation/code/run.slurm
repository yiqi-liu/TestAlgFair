#!/bin/bash
#SBATCH --job-name=simulationDGP
#SBATCH --qos=long
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1     # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=32       # Number of threads per task (OMP threads)

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yl3467@cornell.edu

module load R/4.4.3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R CMD BATCH simulation-DGPplot.R ../results/simulation-DGPplot.txt
