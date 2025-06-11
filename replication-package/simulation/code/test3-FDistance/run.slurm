#!/bin/bash
#SBATCH --job-name=test3n1k
#SBATCH --qos=long
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1
#SBATCH --cluster=cbsugpu09
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1     # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=4       # Number of threads per task (OMP threads)

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yl3467@cornell.edu

module load R/4.4.3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R CMD BATCH test3n1k.R ../../results/test3-FDistance/test3n1k.txt
