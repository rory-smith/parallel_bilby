#!/bin/bash
#
#SBATCH --job-name=GW170817_post
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --output=logs/%A_%a.out

source /home/gashton/.bashrc
conda activate mpi-bilby
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=16
mpirun parallel_bilby_postprocess -r outdir/GW170817_result.json
